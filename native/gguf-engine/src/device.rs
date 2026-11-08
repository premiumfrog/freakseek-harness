//! Compute-device selection and layer placement.
//!
//! The engine runs on the CPU unless it was built with the `cuda` feature and a
//! CUDA device is requested or detected. With a GPU, the last `gpu_layers`
//! transformer layers and the output head run on it; earlier layers and the
//! embedding lookup stay on the CPU, so a model larger than VRAM still loads.

use anyhow::Result;
use candle_core::Device;

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
            "auto" => cuda(0).unwrap_or(Device::Cpu),
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
    }

    #[test]
    fn cpu_offloads_nothing() {
        let p = Placement::cpu();
        assert!(!(0..4).any(|i| p.offloads(i, 4)));
        assert_eq!(p.label(), "cpu");
    }
}
