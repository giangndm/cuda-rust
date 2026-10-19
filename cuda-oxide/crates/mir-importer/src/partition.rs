/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Partitioned and cached device code generation.
//!
//! The monolithic pipeline lowers every collected function of a crate into one
//! `dialect-mir` module, one LLVM module and one `opt`/`llc` invocation. For a
//! crate with hundreds of kernels that is a single-threaded critical path,
//! and any edit recompiles everything.
//!
//! This route splits the crate into kernel groups chosen by the frontend
//! (for example one group per Rust module that defines kernels). Each group
//! holds its kernels plus every device function they reach, so a helper used
//! by two groups is compiled into both, exactly as a separate crate would.
//!
//! ```text
//!  rustc thread (stable MIR context)          worker pool (N threads)
//!  ──────────────────────────────────         ─────────────────────────────
//!  for each group (largest first):
//!    translate group ─► dialect-mir module
//!    key = sha256(printed module, options,
//!                 backend, toolchain, externs)
//!    cache hit?  ── yes ─► reuse PTX
//!         │ no
//!         └──────────── (Context, module) ──► prep ► lower ► export .ll
//!                                             llvm-link ► opt ► llc ► PTX
//!                                             store PTX in the cache
//!  merge group PTX in a fixed order ─► one PTX module (one header,
//!  global `.file` table, per-group `$L__` labels, one block per
//!  debug section, deduplicated module-scope declarations)
//! ```
//!
//! The cache key is computed from the translated `dialect-mir` text, which
//! already contains every resolved type layout, constant and source location
//! the later stages consume, so the key cannot miss a dependency hidden in a
//! type definition or an upstream crate. Translation therefore still runs for
//! every group; only the stages after it are skipped on a hit.
//!
//! Every group gets its own module disambiguator, so counter-named
//! module-scope symbols (`__shared_mem_*`) stay unique in the merged module.
//! A Rust `static` reached from two groups would be duplicated, so that case
//! is reported as unsupported and the caller falls back to the monolithic
//! pipeline.
//!
//! A single group is the monolithic module itself (same module name and
//! disambiguator, no merge), so its PTX is byte-identical to the monolithic
//! pipeline and the cache only adds whole-crate reuse. With several groups,
//! `opt` sees a smaller module and may inline or schedule a shared helper
//! differently, so an entry's PTX is equivalent but not always identical to
//! the monolithic build.

use crate::pipeline::{
    CollectedFunction, CompilationArtifactKind, CompilationResult, DeviceExternDecl,
    KernelLaunchBounds, PipelineConfig, PipelineError, backend_options_for, stderr_pipeline_trace,
    translate_functions,
};
use cuda_oxide_codegen::__private::{
    ModuleArtifactKind, ModulePipelineRequest, OutputFiles, PipelineTrace,
    compile_translated_module,
};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

/// A kernel group compiled as its own module: indices into the collected
/// function list, in collection order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionPartition {
    /// Stable group name (for example the kernels' Rust module path).
    pub name: String,
    /// Indices of the group's kernels and every function they reach.
    pub members: Vec<usize>,
}

/// One group holding every function: the monolithic module, cached whole.
pub fn whole_crate_group(function_count: usize) -> Vec<FunctionPartition> {
    vec![FunctionPartition {
        name: String::new(),
        members: (0..function_count).collect(),
    }]
}

/// Group `function_count` collected functions around kernel `roots`.
///
/// Each group holds its root kernels plus every function reachable from them
/// through `edges` (`(caller, callee)` indices), in collection order; a helper
/// reached from two groups belongs to both. Fewer than two groups, or a
/// function that no root reaches (an edge the collector did not record),
/// yields the whole-crate group instead.
pub fn group_functions(
    function_count: usize,
    edges: &[(usize, usize)],
    roots: BTreeMap<String, Vec<usize>>,
) -> Vec<FunctionPartition> {
    if roots.len() < 2 {
        return whole_crate_group(function_count);
    }
    let mut successors = vec![Vec::new(); function_count];
    for &(caller, callee) in edges {
        successors[caller].push(callee);
    }
    let mut covered = vec![false; function_count];
    let mut groups = Vec::with_capacity(roots.len());
    for (name, kernels) in roots {
        let mut member = vec![false; function_count];
        let mut worklist = kernels;
        while let Some(next) = worklist.pop() {
            if std::mem::replace(&mut member[next], true) {
                continue;
            }
            worklist.extend(successors[next].iter().copied());
        }
        let members: Vec<usize> = (0..function_count).filter(|&i| member[i]).collect();
        for &i in &members {
            covered[i] = true;
        }
        groups.push(FunctionPartition { name, members });
    }
    if covered.iter().all(|&covered| covered) {
        groups
    } else {
        whole_crate_group(function_count)
    }
}

/// Result of [`run_partitioned_pipeline`].
pub enum PartitionedOutcome {
    /// The merged PTX artifact.
    Compiled(CompilationResult),
    /// The crate needs the monolithic pipeline; the reason is for diagnostics.
    Unsupported(String),
}

/// Bump when the key derivation or the cached payload format changes.
const CACHE_FORMAT: &str = "cuda-oxide-partition-v1";

/// Default cache size bound in MiB (`CUDA_OXIDE_PARTITION_CACHE_MB`).
const DEFAULT_CACHE_LIMIT_MB: u64 = 4096;

/// Default bound on concurrent group compilations (`CUDA_OXIDE_CODEGEN_JOBS`
/// overrides it): each running group holds a Pliron context plus `opt`/`llc`
/// processes, about 1-4 GB for the largest groups.
const DEFAULT_MAX_JOBS: usize = 4;

/// After eviction the cache is trimmed to this fraction of its bound.
const CACHE_TRIM_PERCENT: u64 = 80;

/// Environment variables that only control diagnostics or this cache and
/// therefore do not take part in the key.
const KEY_NEUTRAL_ENV: &[&str] = &[
    "CUDA_OXIDE_VERBOSE",
    "CUDA_OXIDE_TIMINGS",
    "CUDA_OXIDE_PARTITION",
    "CUDA_OXIDE_PARTITION_CACHE",
    "CUDA_OXIDE_PARTITION_CACHE_MB",
    "CUDA_OXIDE_CODEGEN_JOBS",
    "CUDA_OXIDE_KEEP_PARTITION_IR",
    "CUDA_OXIDE_PTX_DIR",
    "CUDA_OXIDE_SHOW_RUSTC_MIR",
    "CUDA_OXIDE_DUMP_MIR",
    "CUDA_OXIDE_DUMP_LLVM",
];

struct Job {
    index: usize,
    key: String,
    ctx: Context,
    module: Ptr<Operation>,
}

// SAFETY: a `Context` owns every IR object its `Ptr`s address and has no
// thread affinity: no thread-local state refers to it, and the job moves the
// whole context, so no IR object is ever reachable from two threads.
unsafe impl Send for Job {}

struct GroupPtx {
    ptx: String,
    target: String,
}

/// Compile `partitions` of `functions` as independent modules in parallel,
/// reusing cached PTX for unchanged groups, and merge the result into one PTX
/// artifact at `<output_dir>/<output_name>.ptx`.
pub fn run_partitioned_pipeline(
    functions: &[CollectedFunction],
    partitions: &[FunctionPartition],
    device_externs: &[DeviceExternDecl],
    config: &PipelineConfig,
    known_defs: crate::translator::facts::KnownDefs,
) -> Result<PartitionedOutcome, PipelineError> {
    if config.emit_nvvm_ir {
        return Ok(PartitionedOutcome::Unsupported(
            "NVVM IR output is linked as one module".to_string(),
        ));
    }
    crate::translator::facts::set_known_defs(known_defs);
    std::fs::create_dir_all(&config.output_dir).map_err(export_error)?;
    let work_dir = config
        .output_dir
        .join(".oxide-partitions")
        .join(&config.output_name);
    std::fs::create_dir_all(&work_dir).map_err(export_error)?;

    let base_backend = backend_options_for(config);
    let crate_disambiguator = config.module_disambiguator.unwrap_or(0);
    let cache = PartitionCache::from_env();
    let options_fingerprint = options_fingerprint(config, device_externs);
    let jobs = codegen_jobs();

    // Largest groups first so the longest jobs start earliest.
    let mut order: Vec<usize> = (0..partitions.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(partitions[index].members.len()));

    let mut results: Vec<Option<Result<GroupPtx, PipelineError>>> =
        (0..partitions.len()).map(|_| None).collect();
    let mut kernel_launch_bounds = BTreeMap::new();
    let mut statics_by_group: Vec<BTreeSet<String>> = vec![BTreeSet::new(); partitions.len()];
    let (mut hits, mut misses) = (0usize, 0usize);
    let translate_start = std::time::Instant::now();

    let translation = std::thread::scope(|scope| -> Result<(), PipelineError> {
        let (job_tx, job_rx) = mpsc::sync_channel::<Job>(jobs);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (result_tx, result_rx) = mpsc::channel::<(usize, Result<GroupPtx, PipelineError>)>();
        for _ in 0..jobs {
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let base_backend = &base_backend;
            let work_dir = &work_dir;
            let cache = &cache;
            scope.spawn(move || {
                loop {
                    let next = job_rx.lock().expect("job queue poisoned").recv();
                    let Ok(job) = next else { break };
                    let index = job.index;
                    let mut backend = base_backend.clone();
                    // A single group is the monolithic module: it keeps the
                    // crate's disambiguator, so its PTX is byte-identical.
                    if partitions.len() > 1 {
                        backend.module_disambiguator = Some(group_disambiguator(
                            crate_disambiguator,
                            &partitions[index].name,
                        ));
                    }
                    let result = compile_group(
                        job,
                        &backend,
                        device_externs,
                        config,
                        work_dir,
                        cache.as_ref(),
                    );
                    if result_tx.send((index, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(result_tx);

        let mut translated = Ok(());
        for &index in &order {
            let partition = &partitions[index];
            let mut ctx = Context::new();
            crate::translator::register_dialects(&mut ctx);
            let module_name: pliron::identifier::Identifier = if partitions.len() == 1 {
                config.output_name.clone()
            } else {
                format!("{}_p{index}", config.output_name)
            }
            .try_into()
            .unwrap_or_else(|_| "kernel".try_into().unwrap());
            let module = pliron::builtin::ops::ModuleOp::new(&mut ctx, module_name);
            let module_op = module.get_operation();
            if let Err(error) = translate_functions(
                &mut ctx,
                module_op,
                partition.members.iter().map(|&member| &functions[member]),
                config,
                &mut kernel_launch_bounds,
            ) {
                translated = Err(error);
                break;
            }
            statics_by_group[index] = rust_static_keys(&ctx, module_op);
            let key = group_key(
                &ctx,
                module_op,
                &options_fingerprint,
                if partitions.len() > 1 {
                    group_disambiguator(crate_disambiguator, &partition.name)
                } else {
                    crate_disambiguator
                },
            );
            if let Some(hit) = cache.as_ref().and_then(|cache| cache.load(&key)) {
                hits += 1;
                results[index] = Some(Ok(hit));
                continue;
            }
            misses += 1;
            if std::env::var_os("CUDA_OXIDE_TIMINGS").is_some() {
                eprintln!(
                    "[oxide-partition] group {index} `{}`: {} functions, cache miss",
                    partition.name,
                    partition.members.len()
                );
            }
            if job_tx
                .send(Job {
                    index,
                    key,
                    ctx,
                    module: module_op,
                })
                .is_err()
            {
                break;
            }
        }
        drop(job_tx);
        cuda_oxide_codegen::timing::record(
            "partitions: translate + key (rustc thread)",
            translate_start,
        );
        for (index, result) in result_rx {
            results[index] = Some(result);
        }
        translated
    });
    translation?;
    cuda_oxide_codegen::timing::record("partitions: all groups compiled", translate_start);
    if config.verbose || std::env::var_os("CUDA_OXIDE_TIMINGS").is_some() {
        eprintln!(
            "[oxide-partition] {} groups: {hits} cached, {misses} compiled, {jobs} jobs",
            partitions.len()
        );
    }

    if let Some(shared) = static_shared_between_groups(&statics_by_group) {
        return Ok(PartitionedOutcome::Unsupported(format!(
            "Rust static `{shared}` is reachable from more than one kernel group"
        )));
    }

    let mut groups = Vec::with_capacity(partitions.len());
    for (index, result) in results.into_iter().enumerate() {
        let group = result.ok_or_else(|| {
            PipelineError::PtxGeneration(format!(
                "kernel group `{}` produced no result",
                partitions[index].name
            ))
        })??;
        groups.push(group);
    }

    let merge_start = std::time::Instant::now();
    let target = groups[0].target.clone();
    if let Some(other) = groups.iter().find(|group| group.target != target) {
        return Ok(PartitionedOutcome::Unsupported(format!(
            "kernel groups resolved different targets ({target} and {})",
            other.target
        )));
    }
    let texts: Vec<&str> = groups.iter().map(|group| group.ptx.as_str()).collect();
    let merged = if let [only] = texts.as_slice() {
        only.to_string()
    } else {
        match merge_group_ptx(&texts) {
            Ok(merged) => merged,
            Err(reason) => return Ok(PartitionedOutcome::Unsupported(reason)),
        }
    };
    let ptx_path = config
        .output_dir
        .join(format!("{}.ptx", config.output_name));
    std::fs::write(&ptx_path, merged).map_err(export_error)?;
    cuda_oxide_codegen::timing::record("partitions: merge PTX", merge_start);
    if let Some(cache) = cache.as_ref() {
        cache.evict();
    }

    Ok(PartitionedOutcome::Compiled(CompilationResult {
        artifact_path: ptx_path.clone(),
        artifact_kind: CompilationArtifactKind::Ptx,
        ll_path: work_dir,
        ptx_path,
        target,
        allow_fma_contraction: config.allow_fma_contraction,
        kernel_launch_bounds: kernel_launch_bounds
            .into_iter()
            .collect::<BTreeMap<String, KernelLaunchBounds>>(),
    }))
}

fn export_error(error: std::io::Error) -> PipelineError {
    PipelineError::Export(error.to_string())
}

fn codegen_jobs() -> usize {
    std::env::var("CUDA_OXIDE_CODEGEN_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&jobs| jobs > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|jobs| jobs.get())
                .unwrap_or(1)
                .min(DEFAULT_MAX_JOBS)
        })
}

/// Disambiguator of one group among several (see the module docs).
fn group_disambiguator(crate_disambiguator: u64, group: &str) -> u64 {
    let digest = Sha256::new()
        .chain_update(crate_disambiguator.to_le_bytes())
        .chain_update(group.as_bytes())
        .finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("sha256 has 32 bytes"))
}

/// Run every stage after translation for one group and cache its PTX.
fn compile_group(
    job: Job,
    backend: &cuda_oxide_codegen::__private::BackendOptions,
    device_externs: &[DeviceExternDecl],
    config: &PipelineConfig,
    work_dir: &Path,
    cache: Option<&PartitionCache>,
) -> Result<GroupPtx, PipelineError> {
    let Job {
        index,
        key,
        mut ctx,
        module,
    } = job;
    let ll_path = work_dir.join(format!("p{index}.ll"));
    let ptx_path = work_dir.join(format!("p{index}.ptx"));
    let intermediates = [
        ll_path.clone(),
        work_dir.join(format!("p{index}.linked.ll")),
        work_dir.join(format!("p{index}.linked.opt.ll")),
        work_dir.join(format!("p{index}.opt.ll")),
    ];
    let mut stale = intermediates.to_vec();
    stale.push(ptx_path.clone());
    let request = ModulePipelineRequest::for_rust_pipeline(
        device_externs,
        false,
        backend,
        config.debug_kind,
        OutputFiles {
            llvm_ir: &ll_path,
            ptx: &ptx_path,
            stale_before_export: &stale,
        },
        PipelineTrace {
            verbose: config.verbose,
            dump_mir: false,
            dump_llvm: false,
            sink: Some(stderr_pipeline_trace),
        },
    );
    let started = std::time::Instant::now();
    let generated = compile_translated_module(&mut ctx, module, &request)?;
    drop(ctx);
    if std::env::var_os("CUDA_OXIDE_TIMINGS").is_some() {
        eprintln!(
            "[oxide-partition] group {index} compiled in {:.2}s",
            started.elapsed().as_secs_f64()
        );
    }
    if generated.artifact_kind != ModuleArtifactKind::Ptx {
        return Err(PipelineError::PtxGeneration(
            "partitioned codegen requires PTX output (libdevice IR linking unavailable?)"
                .to_string(),
        ));
    }
    let ptx = std::fs::read_to_string(&ptx_path).map_err(export_error)?;
    if std::env::var_os("CUDA_OXIDE_KEEP_PARTITION_IR").is_none() {
        for path in &intermediates {
            let _ = std::fs::remove_file(path);
        }
    }
    let group = GroupPtx {
        ptx,
        target: generated.target,
    };
    if let Some(cache) = cache {
        cache.store(&key, &group);
    }
    Ok(group)
}

/// `fmt::Write` sink that feeds a hasher, so the printed module is never
/// materialized as one string.
struct HashWriter<'a>(&'a mut Sha256);

impl std::fmt::Write for HashWriter<'_> {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        self.0.update(text.as_bytes());
        Ok(())
    }
}

fn group_key(
    ctx: &Context,
    module: Ptr<Operation>,
    options_fingerprint: &str,
    disambiguator: u64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_FORMAT.as_bytes());
    hasher.update(options_fingerprint.as_bytes());
    hasher.update(disambiguator.to_le_bytes());
    write!(HashWriter(&mut hasher), "{}", module.deref(ctx).disp(ctx))
        .expect("hashing never fails");
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// Everything besides the translated module that changes the PTX: backend
/// options (target, FMA, MIR passes), debug policy, device externs, the
/// backend and compiler binaries, the LLVM tools they select, and every
/// `CUDA_OXIDE_*` variable that is not purely diagnostic.
fn options_fingerprint(config: &PipelineConfig, device_externs: &[DeviceExternDecl]) -> String {
    let mut backend = backend_options_for(config);
    backend.module_disambiguator = None;
    backend.verbose = false;
    let mut env: Vec<(String, String)> = std::env::vars()
        .filter(|(name, _)| name.starts_with("CUDA_OXIDE_"))
        .filter(|(name, _)| !KEY_NEUTRAL_ENV.contains(&name.as_str()))
        .collect();
    env.sort();
    format!(
        "backend={backend:?}\ndebug={:?}\nfma={}\nexterns={device_externs:?}\nenv={env:?}\nbinaries={}\npath={}\n",
        config.debug_kind,
        config.allow_fma_contraction,
        binaries_identity(),
        std::env::var("PATH").unwrap_or_default(),
    )
}

/// Content hash of the loaded codegen backend plus the identity of the rustc
/// binary (whose sysroot provides the default `llc`/`opt`/`llvm-link`) and
/// the CUDA libdevice it links.
fn binaries_identity() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        // The mapping that contains this very function is the backend library.
        let here = binaries_identity as fn() -> &'static str as usize;
        let backend = std::fs::read_to_string("/proc/self/maps")
            .ok()
            .and_then(|maps| {
                maps.lines().find_map(|line| {
                    let mut fields = line.split_whitespace();
                    let (start, end) = fields.next()?.split_once('-')?;
                    let start = usize::from_str_radix(start, 16).ok()?;
                    let end = usize::from_str_radix(end, 16).ok()?;
                    let path = fields.nth(4)?;
                    (start <= here && here < end).then(|| PathBuf::from(path))
                })
            });
        let backend_hash = backend
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .map(|bytes| hex(&Sha256::digest(bytes)))
            // An unidentifiable backend gets a unique key: never a stale hit.
            .unwrap_or_else(|| format!("unknown-{}", std::process::id()));
        let rustc = std::env::current_exe()
            .ok()
            .map(|path| file_identity(&path))
            .unwrap_or_default();
        let libdevice = libnvvm_sys_libdevice_identity();
        format!("backend={backend_hash};rustc={rustc};libdevice={libdevice}")
    })
}

fn libnvvm_sys_libdevice_identity() -> String {
    ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .chain(std::iter::once("/usr/local/cuda".to_string()))
        .map(|root| Path::new(&root).join("nvvm/libdevice/libdevice.10.bc"))
        .find(|path| path.is_file())
        .map(|path| file_identity(&path))
        .unwrap_or_default()
}

fn file_identity(path: &Path) -> String {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let metadata = std::fs::metadata(&canonical).ok();
    let modified = metadata
        .as_ref()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!(
        "{}:{}:{modified}",
        canonical.display(),
        metadata.map(|metadata| metadata.len()).unwrap_or_default()
    )
}

/// Keys of Rust `static` items a group's translated module allocates.
fn rust_static_keys(ctx: &Context, module: Ptr<Operation>) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    collect_static_keys(ctx, module, &mut keys);
    keys
}

fn collect_static_keys(ctx: &Context, op: Ptr<Operation>, keys: &mut BTreeSet<String>) {
    if let Some(global) = Operation::get_op::<dialect_mir::ops::MirGlobalAllocOp>(op, ctx)
        && let Some(key) = global.get_attr_global_key(ctx)
    {
        let key = String::from(key.clone());
        if dialect_mir::ops::rust_static_symbol_from_global_key(&key).is_some() {
            keys.insert(key);
        }
    }
    let op_ref = op.deref(ctx);
    for region in op_ref.regions() {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                collect_static_keys(ctx, child, keys);
            }
        }
    }
}

fn static_shared_between_groups(groups: &[BTreeSet<String>]) -> Option<String> {
    let mut seen = HashSet::new();
    groups
        .iter()
        .flatten()
        .find(|key| !seen.insert(key.as_str()))
        .cloned()
}

/// On-disk PTX cache shared by every project and worktree of the user.
struct PartitionCache {
    dir: PathBuf,
    limit_bytes: u64,
}

impl PartitionCache {
    fn from_env() -> Option<Self> {
        let dir = match std::env::var("CUDA_OXIDE_PARTITION_CACHE") {
            Ok(value) if value == "off" || value == "0" => return None,
            Ok(value) if !value.is_empty() => PathBuf::from(value),
            _ => std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".cache")))?
                .join("cuda-oxide")
                .join("partitions"),
        };
        std::fs::create_dir_all(&dir).ok()?;
        let limit_mb = std::env::var("CUDA_OXIDE_PARTITION_CACHE_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_CACHE_LIMIT_MB);
        Some(Self {
            dir,
            limit_bytes: limit_mb * 1024 * 1024,
        })
    }

    fn load(&self, key: &str) -> Option<GroupPtx> {
        let ptx_path = self.dir.join(format!("{key}.ptx"));
        let target = std::fs::read_to_string(self.dir.join(format!("{key}.target"))).ok()?;
        let ptx = std::fs::read_to_string(&ptx_path).ok()?;
        // Refresh the entry's age for least-recently-used eviction.
        if let Ok(file) = std::fs::File::options().append(true).open(&ptx_path) {
            let _ = file.set_modified(std::time::SystemTime::now());
        }
        Some(GroupPtx {
            ptx,
            target: target.trim().to_string(),
        })
    }

    /// Publish atomically: the `.target` file is written last and is what
    /// [`Self::load`] requires, so a reader never sees a partial entry.
    fn store(&self, key: &str, group: &GroupPtx) {
        let unique = format!("{}.{:?}", std::process::id(), std::thread::current().id());
        let write = |name: String, bytes: &[u8]| -> std::io::Result<()> {
            let temporary = self.dir.join(format!("{name}.tmp.{unique}"));
            std::fs::write(&temporary, bytes)?;
            std::fs::rename(&temporary, self.dir.join(name))
        };
        let _ = write(format!("{key}.ptx"), group.ptx.as_bytes())
            .and_then(|()| write(format!("{key}.target"), group.target.as_bytes()));
    }

    /// Delete the least recently used entries once the cache exceeds its bound.
    fn evict(&self) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "ptx"))
            .filter_map(|entry| {
                let metadata = entry.metadata().ok()?;
                Some((metadata.modified().ok()?, metadata.len(), entry.path()))
            })
            .collect();
        let mut total: u64 = files.iter().map(|(_, size, _)| size).sum();
        if total <= self.limit_bytes {
            return;
        }
        files.sort();
        let goal = self.limit_bytes / 100 * CACHE_TRIM_PERCENT;
        for (_, size, path) in files {
            if total <= goal {
                break;
            }
            let _ = std::fs::remove_file(path.with_extension("target"));
            let _ = std::fs::remove_file(&path);
            total = total.saturating_sub(size);
        }
    }
}

/// Merge independently generated group PTX modules into one module.
///
/// - one `.version` (the highest), `.target` and `.address_size` header;
/// - every `$L__` label renamed per group (`$L__g<k>_`), since labels such as
///   `$L__info_string0` or `$L__func_begin0` are module-scoped;
/// - one global `.file` table, with `.loc` file indices (including
///   `inlined_at`) remapped and identical files shared;
/// - one block per DWARF section with each group's contents behind a
///   `$L__g<k>__debug_<name>` label, and a group's references to the start of
///   a section it emitted rewritten to that label, so each compile unit keeps
///   pointing at its own abbreviation table;
/// - identical module-scope `.extern`/`.global`/`.const` declarations
///   emitted once; a different declaration of an existing symbol, or two
///   definitions of one function, is a conflict and fails the merge.
pub fn merge_group_ptx(groups: &[&str]) -> Result<String, String> {
    let mut version: Option<(u32, u32)> = None;
    let mut target: Option<String> = None;
    let mut address_size: Option<String> = None;
    let mut body = String::new();
    let mut files: Vec<String> = Vec::new();
    let mut file_index: HashMap<String, usize> = HashMap::new();
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut declarations: HashMap<String, String> = HashMap::new();
    let mut emitted_declarations: HashSet<String> = HashSet::new();
    let mut functions: HashSet<String> = HashSet::new();

    for (group, text) in groups.iter().enumerate() {
        let label_prefix = format!("$L__g{group}_");
        let renamed = text.replace("$L__", &label_prefix);

        // First pass: this group's `.file` table and the sections it defines.
        let mut local_files: HashMap<u64, usize> = HashMap::new();
        let mut local_sections: BTreeSet<String> = BTreeSet::new();
        for line in renamed.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = directive_operands(trimmed, ".file") {
                let (index, identity) = rest
                    .split_once(char::is_whitespace)
                    .ok_or_else(|| format!("malformed .file directive: {line}"))?;
                let index: u64 = index
                    .parse()
                    .map_err(|_| format!("malformed .file index: {line}"))?;
                let identity = identity.trim().to_string();
                let global = *file_index.entry(identity.clone()).or_insert_with(|| {
                    files.push(identity);
                    files.len()
                });
                local_files.insert(index, global);
            } else if let Some(rest) = trimmed.strip_prefix(".section")
                && let Some(name) = rest.split_whitespace().next()
            {
                local_sections.insert(name.to_string());
            }
        }

        let mut lines = renamed.lines().peekable();
        while let Some(line) = lines.next() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix(".version") {
                let parsed = parse_version(rest.trim())
                    .ok_or_else(|| format!("malformed .version: {line}"))?;
                version = Some(version.map_or(parsed, |current| current.max(parsed)));
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix(".target") {
                let rest = rest.trim().to_string();
                if let Some(existing) = &target
                    && *existing != rest
                {
                    return Err(format!("groups disagree on .target: {existing} vs {rest}"));
                }
                target = Some(rest);
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix(".address_size") {
                address_size = Some(rest.trim().to_string());
                continue;
            }
            if is_banner_line(line) {
                continue;
            }
            if directive_operands(trimmed, ".file").is_some() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix(".section") {
                let name = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| format!("malformed .section: {line}"))?
                    .to_string();
                let mut contents = String::new();
                let inline = rest.contains('{');
                if !inline {
                    match lines.next() {
                        Some(open) if open.trim() == "{" => {}
                        other => return Err(format!("section {name} without a block: {other:?}")),
                    }
                    loop {
                        let Some(inner) = lines.next() else {
                            return Err(format!("unterminated section {name}"));
                        };
                        if inner.trim() == "}" {
                            break;
                        }
                        contents.push_str(&rewrite_section_refs(inner, group, &local_sections));
                        contents.push('\n');
                    }
                }
                let block = sections.iter_mut().find(|(existing, _)| *existing == name);
                let block = match block {
                    Some((_, block)) => block,
                    None => {
                        sections.push((name.clone(), String::new()));
                        &mut sections.last_mut().expect("just pushed").1
                    }
                };
                if !contents.is_empty() {
                    let _ = writeln!(block, "{}", section_label(group, &name));
                    block.push_str(&contents);
                }
                continue;
            }
            if directive_operands(trimmed, ".loc").is_some() {
                body.push_str(&remap_loc(line, &local_files)?);
                body.push('\n');
                continue;
            }
            if line.starts_with(".extern")
                || line.starts_with(".global")
                || line.starts_with(".const")
                || line.starts_with(".shared")
            {
                let symbol = declared_symbol(line)
                    .ok_or_else(|| format!("unrecognized declaration: {line}"))?;
                match declarations.get(&symbol) {
                    Some(existing) if existing == line => continue,
                    Some(existing) => {
                        return Err(format!(
                            "conflicting declarations of `{symbol}`: `{existing}` and `{line}`"
                        ));
                    }
                    None => {
                        declarations.insert(symbol, line.to_string());
                        emitted_declarations.insert(line.to_string());
                    }
                }
            }
            if let Some(name) = function_definition_name(line)
                && !functions.insert(name.clone())
            {
                return Err(format!("function `{name}` is defined by two kernel groups"));
            }
            body.push_str(line);
            body.push('\n');
        }
    }

    let (major, minor) = version.ok_or("no .version directive")?;
    let mut merged = String::with_capacity(body.len() + 1024);
    let _ = writeln!(merged, "//\n// Generated by LLVM NVPTX Back-End\n//\n");
    let _ = writeln!(merged, ".version {major}.{minor}");
    let _ = writeln!(merged, ".target {}", target.ok_or("no .target directive")?);
    let _ = writeln!(
        merged,
        ".address_size {}",
        address_size.ok_or("no .address_size directive")?
    );
    merged.push_str(&body);
    for (index, identity) in files.iter().enumerate() {
        let _ = writeln!(merged, "\t.file\t{} {identity}", index + 1);
    }
    for (name, contents) in &sections {
        if contents.is_empty() {
            let _ = writeln!(merged, "\t.section\t{name}\t{{\t}}");
        } else {
            let _ = writeln!(merged, "\t.section\t{name}\n\t{{\n{contents}\t}}");
        }
    }
    Ok(merged)
}

/// Operands of `directive` when `line` is that directive (not merely a
/// directive sharing its prefix, such as `.local` for `.loc`).
fn directive_operands<'a>(line: &'a str, directive: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(directive)?;
    rest.starts_with(char::is_whitespace)
        .then(|| rest.trim_start())
}

/// The LLVM banner; the merged module writes its own.
fn is_banner_line(line: &str) -> bool {
    line == "//" || line == "// Generated by LLVM NVPTX Back-End"
}

fn parse_version(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.split_once('.')?;
    Some((major.trim().parse().ok()?, minor.trim().parse().ok()?))
}

fn section_label(group: usize, section: &str) -> String {
    format!("$L__g{group}_{}:", section.replace('.', "_"))
}

/// Point `.b32 .debug_<name>` at this group's block of a section it emitted.
fn rewrite_section_refs(line: &str, group: usize, local_sections: &BTreeSet<String>) -> String {
    let trimmed = line.trim_start();
    for width in [".b32", ".b64"] {
        if let Some(rest) = trimmed.strip_prefix(width) {
            let operand = rest.trim_start();
            let name: String = operand.chars().take_while(|c| !c.is_whitespace()).collect();
            if local_sections.contains(&name) {
                let label = section_label(group, &name);
                let label = label.trim_end_matches(':');
                return line.replacen(&name, label, 1);
            }
        }
    }
    line.to_string()
}

/// Rewrite the file indices of `.loc F L C[, function_name X, inlined_at F L C]`.
fn remap_loc(line: &str, files: &HashMap<u64, usize>) -> Result<String, String> {
    let (code, comment) = match line.find("//") {
        Some(position) => line.split_at(position),
        None => (line, ""),
    };
    let lead_len = code.len() - code.trim_start().len();
    let mut out = String::with_capacity(line.len());
    out.push_str(&code[..lead_len]);
    let mut remap_next = true;
    let mut first = true;
    for token in code[lead_len..].split_inclusive(|c: char| c.is_whitespace() || c == ',') {
        let word = token.trim_end_matches(|c: char| c.is_whitespace() || c == ',');
        if first {
            first = false;
            out.push_str(token);
            continue;
        }
        if remap_next && !word.is_empty() {
            let index: u64 = word
                .parse()
                .map_err(|_| format!("malformed .loc file index: {line}"))?;
            let global = files
                .get(&index)
                .ok_or_else(|| format!(".loc references unknown file {index}: {line}"))?;
            out.push_str(&token.replacen(word, &global.to_string(), 1));
            remap_next = false;
            continue;
        }
        if word == "inlined_at" {
            remap_next = true;
        }
        out.push_str(token);
    }
    out.push_str(comment);
    Ok(out)
}

/// Symbol named by a module-scope declaration such as
/// `.global .align 4 .b8 name[24] = {...};` or `.extern .shared .b8 name[];`.
fn declared_symbol(line: &str) -> Option<String> {
    let head = line.split(['=', ';']).next()?;
    let head = head.split('[').next()?.trim_end();
    let symbol = head.rsplit(char::is_whitespace).next()?;
    (!symbol.is_empty()).then(|| symbol.to_string())
}

/// Name of a function or kernel defined (not declared) on this line.
fn function_definition_name(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix(".visible .entry ")
        .or_else(|| line.strip_prefix(".entry "))
        .or_else(|| line.strip_prefix(".visible .func "))
        .or_else(|| line.strip_prefix(".func "))
        .or_else(|| line.strip_prefix(".weak .func "))?;
    // `.func (.param ...) name(` puts the return value first.
    let rest = if rest.starts_with('(') {
        rest.split_once(')')?.1.trim_start()
    } else {
        rest
    };
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
        .collect();
    (!name.is_empty() && !line.trim_end().ends_with(';')).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUP_A: &str = "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 8.7\n.target sm_120a\n.address_size 64\n\n.global .align 4 .b8 table[2] = {1, 2};\n.visible .entry a(\n)\n{\n$L__func_begin0:\n\t.loc\t1 3 4\n\t.loc\t2 5 6, function_name $L__info_string0, inlined_at 1 7 8\n\tret;\n}\n\t.file\t1 \"a.rs\"\n\t.file\t2 \"common.rs\"\n\t.section\t.debug_abbrev\n\t{\n.b8 1\n\t}\n\t.section\t.debug_info\n\t{\n.b32 .debug_abbrev\n.b32 .debug_line\n.b32 $L__info_string0\n\t}\n\t.section\t.debug_str\n\t{\n$L__info_string0:\n.b8 0\n\t}\n\t.section\t.debug_macinfo\t{\t}\n";
    const GROUP_B: &str = "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 8.8\n.target sm_120a\n.address_size 64\n\n.global .align 4 .b8 table[2] = {1, 2};\n.visible .entry b(\n)\n{\n$L__func_begin0:\n\t.loc\t1 9 9\n\tret;\n}\n\t.file\t1 \"common.rs\"\n\t.section\t.debug_abbrev\n\t{\n.b8 1\n\t}\n\t.section\t.debug_info\n\t{\n.b32 .debug_abbrev\n\t}\n\t.section\t.debug_macinfo\t{\t}\n";

    #[test]
    fn merges_headers_files_labels_and_sections() {
        let merged = merge_group_ptx(&[GROUP_A, GROUP_B]).unwrap();
        assert_eq!(merged.matches(".version").count(), 1);
        assert!(merged.contains(".version 8.8"));
        assert_eq!(merged.matches(".global .align 4 .b8 table").count(), 1);
        assert!(merged.contains("$L__g0_func_begin0:"));
        assert!(merged.contains("$L__g1_func_begin0:"));
        assert!(merged.contains("\t.loc\t1 3 4"));
        assert!(
            merged.contains("\t.loc\t2 5 6, function_name $L__g0_info_string0, inlined_at 1 7 8")
        );
        // `common.rs` is file 2 globally; group B's file 1 maps to it.
        assert!(merged.contains("\t.loc\t2 9 9"));
        assert_eq!(merged.matches(".file").count(), 2);
        assert_eq!(merged.matches(".section\t.debug_abbrev").count(), 1);
        assert!(merged.contains(".b32 $L__g0__debug_abbrev"));
        assert!(merged.contains(".b32 $L__g1__debug_abbrev"));
        assert!(merged.contains(".b32 .debug_line"));
        assert!(merged.contains("\t.section\t.debug_macinfo\t{\t}"));
    }

    #[test]
    fn conflicting_declarations_fail() {
        let other = GROUP_B.replace("{1, 2}", "{3, 4}");
        assert!(merge_group_ptx(&[GROUP_A, &other]).is_err());
    }

    #[test]
    fn duplicate_kernels_fail() {
        assert!(merge_group_ptx(&[GROUP_A, GROUP_A]).is_err());
    }

    #[test]
    fn local_depot_is_not_a_loc_directive() {
        let local = GROUP_B.replace(
            "\tret;",
            "\t.local .align 4 .b8 \t__local_depot0[4];\n\tret;",
        );
        let merged = merge_group_ptx(&[GROUP_A, &local]).unwrap();
        assert!(merged.contains("\t.local .align 4 .b8 \t__local_depot0[4];"));
    }

    #[test]
    fn merged_module_has_one_header_and_deduplicated_globals() {
        let merged = merge_group_ptx(&[GROUP_A, GROUP_B]).unwrap();
        for directive in [".version", ".target", ".address_size", "Generated by LLVM"] {
            assert_eq!(merged.matches(directive).count(), 1, "{directive}");
        }
        assert_eq!(merged.matches("table[2]").count(), 1);
        assert_eq!(merged.matches(".entry").count(), 2);
    }

    fn btree(groups: &[(&str, &[usize])]) -> BTreeMap<String, Vec<usize>> {
        groups
            .iter()
            .map(|(name, roots)| (name.to_string(), roots.to_vec()))
            .collect()
    }

    #[test]
    fn groups_hold_their_kernels_and_every_reachable_helper() {
        // 0 = kernel a, 1 = kernel b, 2 = shared helper, 3 = helper of 2,
        // 4 = kernel c, 5 = private helper of a.
        let edges = [(0, 2), (1, 2), (2, 3), (0, 5)];
        let groups = group_functions(6, &edges, btree(&[("m1", &[0]), ("m2", &[1, 4])]));
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].name, "m1");
        assert_eq!(groups[0].members, vec![0, 2, 3, 5]);
        assert_eq!(groups[1].members, vec![1, 2, 3, 4]);
    }

    #[test]
    fn unreached_function_or_single_group_selects_the_whole_crate() {
        let whole = whole_crate_group(4);
        assert_eq!(
            group_functions(4, &[(0, 1)], btree(&[("m1", &[0]), ("m2", &[2])])),
            whole
        );
        assert_eq!(
            group_functions(4, &[], btree(&[("m1", &[0, 1, 2, 3])])),
            whole
        );
    }

    /// A module with one `MirFuncOp` per `(name, callees)`, each body made of
    /// calls to its callees.
    fn module_with(ctx: &mut Context, functions: &[(&str, &[&str])]) -> Ptr<Operation> {
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::attributes::{StringAttr, TypeAttr};
        use pliron::builtin::op_interfaces::SymbolOpInterface;
        use pliron::builtin::types::FunctionType;

        crate::translator::register_dialects(ctx);
        let module = pliron::builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let module_op = module.get_operation();
        for (name, callees) in functions {
            let function_type = FunctionType::get(ctx, vec![], vec![]);
            let function_op = Operation::new(
                ctx,
                dialect_mir::ops::MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let function = dialect_mir::ops::MirFuncOp::new(
                ctx,
                function_op,
                TypeAttr::new(function_type.into()),
            );
            function.set_symbol_name(ctx, (*name).try_into().unwrap());
            let body = function_op.deref(ctx).get_region(0);
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(body, ctx);
            for callee in *callees {
                let call_op = Operation::new(
                    ctx,
                    dialect_mir::ops::MirCallOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![],
                    0,
                );
                dialect_mir::ops::MirCallOp::new(call_op)
                    .set_attr_callee(ctx, StringAttr::new((*callee).to_string()));
                call_op.insert_at_back(block, ctx);
            }
            cuda_oxide_codegen::__private::append_to_module(ctx, module_op, function_op);
        }
        module_op
    }

    /// Keys of groups `a = {kernel_a, helper}`, `b = {kernel_b, helper}` and
    /// `c = {kernel_c}`, each built in its own context like the pipeline does.
    fn group_keys(kernel_a_body: &[&str], helper_body: &[&str]) -> [String; 3] {
        let groups: [&[(&str, &[&str])]; 3] = [
            &[("kernel_a", kernel_a_body), ("helper", helper_body)],
            &[("kernel_b", &["helper"]), ("helper", helper_body)],
            &[("kernel_c", &["leaf"])],
        ];
        groups.map(|functions| {
            let mut ctx = Context::new();
            let module = module_with(&mut ctx, functions);
            group_key(&ctx, module, "options", 7)
        })
    }

    #[test]
    fn group_key_is_deterministic_across_contexts() {
        assert_eq!(
            group_keys(&["helper"], &["leaf"]),
            group_keys(&["helper"], &["leaf"])
        );
    }

    #[test]
    fn kernel_change_only_changes_its_group_key() {
        let before = group_keys(&["helper"], &["leaf"]);
        let after = group_keys(&["helper", "extra"], &["leaf"]);
        assert_ne!(before[0], after[0]);
        assert_eq!(before[1], after[1]);
        assert_eq!(before[2], after[2]);
    }

    #[test]
    fn shared_helper_change_changes_every_dependent_group_key() {
        let before = group_keys(&["helper"], &["leaf"]);
        let after = group_keys(&["helper"], &["other_leaf"]);
        assert_ne!(before[0], after[0]);
        assert_ne!(before[1], after[1]);
        assert_eq!(before[2], after[2]);
    }

    #[test]
    fn group_key_covers_options_and_disambiguator() {
        let key = |options: &str, disambiguator: u64| {
            let mut ctx = Context::new();
            let module = module_with(&mut ctx, &[("kernel", &[])]);
            group_key(&ctx, module, options, disambiguator)
        };
        assert_ne!(key("sm_120a", 1), key("sm_90", 1));
        assert_ne!(key("sm_120a", 1), key("sm_120a", 2));
    }

    fn scratch_cache(limit_bytes: u64) -> PartitionCache {
        let dir = std::env::temp_dir().join(format!(
            "oxide-partition-cache-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        PartitionCache { dir, limit_bytes }
    }

    #[test]
    fn cache_round_trips_ptx_and_target() {
        let cache = scratch_cache(u64::MAX);
        assert!(cache.load("k1").is_none());
        cache.store(
            "k1",
            &GroupPtx {
                ptx: GROUP_A.to_string(),
                target: "sm_120a".to_string(),
            },
        );
        let loaded = cache.load("k1").unwrap();
        assert_eq!(loaded.ptx, GROUP_A);
        assert_eq!(loaded.target, "sm_120a");
        let _ = std::fs::remove_dir_all(&cache.dir);
    }

    #[test]
    fn cache_eviction_drops_least_recently_used_entries() {
        let cache = scratch_cache(2500);
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        for (index, key) in ["oldest", "middle", "newest"].iter().enumerate() {
            let group = GroupPtx {
                ptx: "x".repeat(1000),
                target: "sm_120a".to_string(),
            };
            cache.store(key, &group);
            let file = std::fs::File::options()
                .append(true)
                .open(cache.dir.join(format!("{key}.ptx")))
                .unwrap();
            file.set_modified(old + std::time::Duration::from_secs(index as u64 * 60))
                .unwrap();
        }
        // 3000 bytes over a 2500-byte bound: trim to 80% (2000) by dropping
        // only the least recently used entry.
        cache.evict();
        assert!(cache.load("oldest").is_none());
        assert!(cache.load("middle").is_some());
        assert!(cache.load("newest").is_some());
        let _ = std::fs::remove_dir_all(&cache.dir);
    }
}
