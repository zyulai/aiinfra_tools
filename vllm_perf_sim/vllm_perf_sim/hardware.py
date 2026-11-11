"""Public hardware specifications and explicit, user-editable performance assumptions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    memory_gb: float
    bandwidth_gbs: float
    compute_tflops: dict[str, float]
    nvlink_gbs: float = 0.0
    interconnect_gbs: float = 25.0
    flops_efficiency: float = 0.45
    memory_efficiency: float = 0.75
    base_step_overhead_s: float = 0.00015
    per_sequence_overhead_s: float = 0.000012
    spec_note: str = ""


# Vendor specifications, not measured vLLM performance. Efficiency defaults are
# deliberately exposed assumptions and should be calibrated with real runs.
HARDWARE = {
    "h100-sxm": HardwareSpec(
        "H100 SXM 80GB", 80, 3350, {"bf16": 989.4, "fp16": 989.4, "fp8": 1979.0},
        nvlink_gbs=450, flops_efficiency=0.45, memory_efficiency=0.75,
        base_step_overhead_s=0.00012, per_sequence_overhead_s=0.000008,
        spec_note="989.4 TFLOPS dense BF16 with sparsity excluded; 3.35 TB/s HBM3",
    ),
    "h100-pcie": HardwareSpec(
        "H100 PCIe 80GB", 80, 2000, {"bf16": 756.0, "fp16": 756.0, "fp8": 1513.0},
        nvlink_gbs=300, flops_efficiency=0.42, memory_efficiency=0.72,
    ),
    "h200": HardwareSpec(
        "H200 SXM 141GB", 141, 4800, {"bf16": 989.4, "fp16": 989.4, "fp8": 1979.0},
        nvlink_gbs=450, flops_efficiency=0.45, memory_efficiency=0.75,
    ),
    "a100-80": HardwareSpec(
        "A100 SXM 80GB", 80, 2039, {"bf16": 312.0, "fp16": 312.0},
        nvlink_gbs=300, flops_efficiency=0.45, memory_efficiency=0.72,
        base_step_overhead_s=0.00018, per_sequence_overhead_s=0.000012,
    ),
    "a100-40": HardwareSpec(
        "A100 SXM 40GB", 40, 1555, {"bf16": 312.0, "fp16": 312.0},
        nvlink_gbs=300, flops_efficiency=0.43, memory_efficiency=0.70,
        base_step_overhead_s=0.00020, per_sequence_overhead_s=0.000014,
    ),
    "l40s": HardwareSpec(
        "L40S 48GB", 48, 864, {"bf16": 362.05, "fp16": 362.05, "fp8": 733.0},
        nvlink_gbs=0, interconnect_gbs=32, flops_efficiency=0.40,
        memory_efficiency=0.70, base_step_overhead_s=0.00022,
    ),
    "rtx4090": HardwareSpec(
        "RTX 4090 24GB", 24, 1008, {"bf16": 165.2, "fp16": 165.2, "fp8": 330.3},
        nvlink_gbs=0, interconnect_gbs=32, flops_efficiency=0.35,
        memory_efficiency=0.68, base_step_overhead_s=0.00025,
    ),
    "rtx5090": HardwareSpec(
        "RTX 5090 32GB", 32, 1792, {"bf16": 209.5, "fp16": 209.5, "fp8": 419.0},
        nvlink_gbs=0, interconnect_gbs=32, flops_efficiency=0.38,
        memory_efficiency=0.70, base_step_overhead_s=0.00022,
    ),
}


def get_hardware(name: str, **overrides) -> HardwareSpec:
    try:
        spec = HARDWARE[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown GPU {name!r}; choose from {', '.join(HARDWARE)}") from exc
    unknown = set(overrides) - set(spec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown hardware override(s): {', '.join(sorted(unknown))}")
    values = {key: value for key, value in overrides.items() if value is not None}
    if values:
        import dataclasses
        spec = dataclasses.replace(spec, **values)
    return spec
