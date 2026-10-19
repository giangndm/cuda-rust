//! Opt-in stage timing for the device pipeline (`CUDA_OXIDE_TIMINGS=1`).

use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("CUDA_OXIDE_TIMINGS").is_some())
}

/// Print the time elapsed since `start` for `stage` when timings are enabled.
pub fn record(stage: &str, start: Instant) {
    if enabled() {
        eprintln!(
            "[oxide-timing] {stage}: {:.2}s",
            start.elapsed().as_secs_f64()
        );
    }
}
