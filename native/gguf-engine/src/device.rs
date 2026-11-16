//! Compute-device selection and layer placement.
//!
//! The engine runs on the CPU unless it was built with the `cuda` feature and a
//! CUDA device is requested or detected. With a GPU, the last `gpu_layers`
//! transformer layers and the output head run on it; earlier layers and the
//! embedding lookup stay on the CPU, so a model larger than VRAM still loads.

use anyhow::Result;
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use serde::Serialize;

use crate::engine::EngineError;

/// Accelerator memory left free for KV caches, activations, and the kernels' scratch buffers.
const VRAM_RESERVE: u64 = 1536 << 20;

/// Where a model's layers run.
#[derive(Debug, Clone)]
pub struct Placement {
    /// Accelerator for offloaded layers; `Device::Cpu` when none.
    pub device: Device,
    /// Trailing layers placed on `device`; `None` places every layer there.
    pub gpu_layers: Option<usize>,
    /// Accelerator bytes the model may use; `None` places no limit. A model with routed experts
    /// and no `gpu_layers` keeps the experts that exceed it in system memory.
    pub vram_budget: Option<u64>,
}

impl Placement {
    /// Every layer on the CPU.
    pub fn cpu() -> Self {
        Self { device: Device::Cpu, gpu_layers: None, vram_budget: None }
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
        let vram_budget = match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => {
                devices().ok().and_then(|d| d.into_iter().find(|d| d.ordinal == gpu_id)).map(|d| d.free_bytes.saturating_sub(VRAM_RESERVE))
            }
            _ => None,
        };
        Ok(Self { device, gpu_layers, vram_budget })
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

/// Largest relative error accepted between an accelerator's quantized product and the host's exact one.
/// The accelerator kernels round activations to 8 bits, which stays well below this.
const KERNEL_TOLERANCE: f32 = 0.08;

/// Compare the accelerator's quantized matrix products with exact host products.
/// @param dev - accelerator to check.
/// @param dtypes - weight types to check; a type candle cannot quantize from floats is skipped.
/// @returns per checked type, the worst error relative to the largest reference value.
pub fn verify_quantized(dev: &Device, dtypes: &[GgmlDType]) -> Result<Vec<(GgmlDType, f32)>> {
    const ROWS: usize = 64;
    const COLS: usize = 512;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut noise = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let weights = Tensor::from_vec(noise(ROWS * COLS), (ROWS, COLS), &Device::Cpu)?;
    // One row, a small batch, and a prefill-sized batch take different kernels.
    let inputs = [1usize, 6, 48].map(|b| Tensor::from_vec(noise(b * COLS), (b, COLS), &Device::Cpu));
    let mut out = Vec::new();
    for &dtype in dtypes {
        let Ok(host) = QTensor::quantize(&weights, dtype) else { continue };
        let exact = host.dequantize(&Device::Cpu)?.t()?;
        let on_dev = QMatMul::from_qtensor(qtensor_from_ggml(dtype, &host.data()?, vec![ROWS, COLS], dev)?)?;
        let mut worst = 0f32;
        for x in &inputs {
            let x = x.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?;
            let want = x.matmul(&exact)?.flatten_all()?.to_vec1::<f32>()?;
            let got = on_dev.forward(&x.to_device(dev)?)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let diff = got.iter().zip(&want).map(|(a, b)| if a.is_finite() { (a - b).abs() } else { f32::INFINITY }).fold(0f32, f32::max);
            worst = worst.max(diff / scale);
        }
        out.push((dtype, worst));
    }
    Ok(out)
}

/// Check the accelerator's quantized kernels for a model's weight types before trusting them.
/// A CUDA device whose batched kernels disagree with the host is retried with candle's
/// per-row kernels; if those disagree too, loading on this device fails.
/// @param dev - accelerator; the host passes without a check.
/// @param dtypes - weight types the model uses.
pub fn ensure_kernels(dev: &Device, dtypes: &[GgmlDType]) -> Result<()> {
    if dev.is_cpu() {
        return Ok(());
    }
    let failing = |checks: &[(GgmlDType, f32)]| -> Vec<String> {
        checks.iter().filter(|c| !(c.1 <= KERNEL_TOLERANCE)).map(|c| format!("{:?} (error {:.2})", c.0, c.1)).collect()
    };
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut bad = failing(&verify_quantized(dev, dtypes)?);
    #[cfg(feature = "cuda")]
    if !bad.is_empty() {
        candle_core::quantized::cuda::set_force_dmmv(true);
        bad = failing(&verify_quantized(dev, dtypes)?);
    }
    if bad.is_empty() {
        return Ok(());
    }
    Err(EngineError { code: "GPU_KERNEL_MISMATCH", message: format!("this GPU computes wrong results for {}; update the NVIDIA driver or load on the CPU", bad.join(", ")) }.into())
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
    fn host_kernels_match_exact_products() {
        let checks = verify_quantized(&Device::Cpu, &[GgmlDType::Q8_0, GgmlDType::Q4_0, GgmlDType::Q3K, GgmlDType::Q5K, GgmlDType::F16]).unwrap();
        assert!(!checks.is_empty());
        for (dtype, error) in checks {
            assert!(error < 0.03, "{dtype:?}: {error}");
        }
        ensure_kernels(&Device::Cpu, &[GgmlDType::Q8_0]).unwrap();
    }

    #[test]
    fn cpu_offloads_nothing() {
        let p = Placement::cpu();
        assert!(!(0..4).any(|i| p.offloads(i, 4)));
        assert_eq!(p.label(), "cpu");
    }
}
