//! Compute-device selection and layer placement.
//!
//! The engine runs on the CPU unless it was built with the `cuda` feature and a
//! CUDA device is requested or detected. With a GPU, the last `gpu_layers`
//! transformer layers and the output head run on it; earlier layers and the
//! embedding lookup stay on the CPU, so a model larger than VRAM still loads.

use anyhow::Result;
use candle_core::Device;
use serde::Serialize;

use crate::engine::EngineError;

/// Where a model's layers run.
#[derive(Debug, Clone)]
pub struct Placement {
    /// Accelerator for offloaded layers; `Device::Cpu` when none.
    pub device: Device,
    /// Trailing layers placed on `device`; `None` places every layer there.
    pub gpu_layers: Option<usize>,
}

impl Placement {
    /// Every layer on the CPU.
    pub fn cpu() -> Self {
        Self { device: Device::Cpu, gpu_layers: None }
    }

    /// Resolve a wire device request.
    /// @param spec - `auto`, `cpu`, `cuda`, or `cuda:<ordinal>`; `None` means `auto`.
    /// @param gpu_layers - trailing layers to offload; `None` offloads all.
    pub fn resolve(spec: Option<&str>, gpu_layers: Option<usize>) -> Result<Self> {
        let spec = spec.unwrap_or("auto");
        let device = match spec {
            "cpu" => Device::Cpu,
            // `auto` skips a GPU older than the kernels' compute capability, which could open but not run them.
            "auto" => match devices().ok().and_then(|d| d.into_iter().next()) {
                Some(d) if d.supported => cuda(0).unwrap_or(Device::Cpu),
                _ => Device::Cpu,
            },
            "cuda" => cuda(0)?,
            other => match other.strip_prefix("cuda:").and_then(|n| n.parse::<usize>().ok()) {
                Some(ordinal) => cuda(ordinal)?,
                None => {
                    return Err(EngineError { code: "BAD_DEVICE", message: format!("unknown device `{other}`; expected auto, cpu, cuda, or cuda:<n>") }.into())
                }
            },
        };
        let gpu_layers = if device.is_cpu() { None } else { gpu_layers };
        Ok(Self { device, gpu_layers })
    }

    /// Whether layer `index` of `n_layer` runs on the accelerator.
    /// @param index - zero-based layer index.
    /// @param n_layer - total layers.
    pub fn offloads(&self, index: usize, n_layer: usize) -> bool {
        !self.device.is_cpu() && index + self.gpu_layers.unwrap_or(n_layer).min(n_layer) >= n_layer
    }

    /// Wire label of the accelerator, e.g. `cpu` or `cuda:0`.
    pub fn label(&self) -> String {
        match self.device.location() {
            candle_core::DeviceLocation::Cpu => "cpu".into(),
            candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
        }
    }
}

/// Whether this build can run on CUDA devices.
pub fn cuda_compiled() -> bool {
    cfg!(feature = "cuda")
}

/// Lowest compute capability the CUDA kernels were built for, as `major * 10 + minor`.
/// The build's `CUDA_COMPUTE_CAP` sets it; FreakSeek releases build for 75 (RTX 20-series), and newer
/// GPUs compile the kernels' PTX on first use.
pub fn min_compute_capability() -> u32 {
    option_env!("CUDA_COMPUTE_CAP").and_then(|v| v.trim().parse().ok()).unwrap_or(75)
}

/// One CUDA device as the driver reports it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// `major.minor`, for example `8.6`.
    pub compute_capability: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// The device meets [`min_compute_capability`].
    pub supported: bool,
}

/// CUDA devices visible to this build; empty for a CPU build or when the driver reports none.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::CudaContext;
        let Ok(count) = CudaContext::device_count() else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        for ordinal in 0..usize::try_from(count).unwrap_or(0) {
            let ctx = CudaContext::new(ordinal).map_err(|e| anyhow::anyhow!("CUDA device {ordinal}: {e:?}"))?;
            let (major, minor) = ctx.compute_capability().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let (free, total) = ctx.mem_get_info().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            out.push(DeviceInfo {
                ordinal,
                name: ctx.name().map_err(|e| anyhow::anyhow!("{e:?}"))?,
                compute_capability: format!("{major}.{minor}"),
                total_bytes: total as u64,
                free_bytes: free as u64,
                supported: (major * 10 + minor) as u32 >= min_compute_capability(),
            });
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    Ok(Vec::new())
}

fn cuda(ordinal: usize) -> Result<Device> {
    if !cuda_compiled() {
        return Err(EngineError { code: "DEVICE_UNAVAILABLE", message: "this engine was built without CUDA; rebuild with `--features cuda`".into() }.into());
    }
    Device::new_cuda(ordinal).map_err(|e| EngineError { code: "DEVICE_UNAVAILABLE", message: format!("CUDA device {ordinal} unavailable: {}", e.to_string().lines().next().unwrap_or_default()) }.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_and_auto_resolve_without_cuda() {
        assert!(Placement::resolve(Some("cpu"), Some(3)).unwrap().device.is_cpu());
        assert_eq!(Placement::resolve(Some("cpu"), Some(3)).unwrap().gpu_layers, None);
        if !cuda_compiled() {
            assert!(Placement::resolve(None, None).unwrap().device.is_cpu());
            assert!(Placement::resolve(Some("cuda"), None).is_err());
        }
        assert!(Placement::resolve(Some("tpu"), None).is_err());
        if !cuda_compiled() {
            assert!(devices().unwrap().is_empty());
        }
    }

    #[test]
    fn cpu_offloads_nothing() {
        let p = Placement::cpu();
        assert!(!(0..4).any(|i| p.offloads(i, 4)));
        assert_eq!(p.label(), "cpu");
    }
}
