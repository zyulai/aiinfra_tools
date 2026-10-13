"""
Table renderer — format parallel analysis results as rich tables.
Falls back to plain text if `rich` is not installed.
"""
import os

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .parallel_engine import ShardedKernel, ParallelConfig
from .config_parser import ModelConfig


def _fmt_shape(shape: tuple) -> str:
    """Format a weight/activation shape tuple."""
    if not shape or shape == (0,):
        return "—"
    return "×".join(str(d) for d in shape)


def _fmt_bytes(n: int) -> str:
    """Format bytes as human-readable."""
    if n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_shape_diff(orig: tuple, shard: tuple) -> str:
    """Show original → sharded with coloring."""
    s_orig = _fmt_shape(orig)
    s_shard = _fmt_shape(shard)
    if s_orig == s_shard:
        return s_orig
    return f"{s_orig} → {s_shard}"


def render_kernel_table_rich(
    cfg: ModelConfig,
    pc: ParallelConfig,
    results: list[ShardedKernel],
) -> str:
    """Render full kernel sharding table using rich."""
    console = Console(record=True, width=200, force_terminal=True)
    
    # ── Header panel ──
    header = f"""Model: {cfg.model_type} | {cfg.architectures}
  Hidden={cfg.hidden_size}  Layers={cfg.num_hidden_layers}  Heads={cfg.num_attention_heads}  KV={cfg.kv_heads}  HeadDim={cfg.effective_head_dim}
  Vocab={cfg.vocab_size}  Inter={cfg.intermediate_size}  MoE={cfg.is_moe}{'  Experts='+str(cfg.num_experts) if cfg.is_moe else ''}
Parallel: {pc}
World Size: {pc.world_size} GPU(s)"""

    console.print(Panel(header, title="Parallel Tool — Kernel Sharding Analysis", 
                        border_style="bright_blue", expand=False))

    # ── PP stage summary ──
    if pc.pp > 1:
        console.print()
        pp_table = Table(title="Pipeline Parallel — Stage Layout", box=box.ROUNDED, show_lines=True)
        pp_table.add_column("Stage", style="bold cyan", justify="center")
        pp_table.add_column("Layer Range", style="white")
        pp_table.add_column("Layers", justify="center")
        
        layers_per_stage = cfg.num_hidden_layers // pc.pp
        extra = cfg.num_hidden_layers % pc.pp
        cumulative = 0
        for s in range(pc.pp):
            stage_layers = layers_per_stage + (1 if s < extra else 0)
            end = cumulative + stage_layers - 1
            pp_table.add_row(f"{s}", f"[{cumulative}] → [{end}]", f"{stage_layers}")
            cumulative += stage_layers
        console.print(pp_table)

    # ── Per-category kernel tables ──
    categories = ["embedding", "attention", "mlp", "moe", "norm", "lm_head"]
    cat_names = {
        "embedding": "Embedding",
        "attention": "Attention (Q/KV/O + Core)",
        "mlp": "MLP / FFN (Gate/Up/Down)",
        "moe": "MoE (Router + Experts)",
        "norm": "LayerNorm / RMSNorm",
        "lm_head": "LM Head",
    }

    for cat in categories:
        cat_results = [r for r in results if r.kernel.category == cat]
        if not cat_results:
            continue

        table = Table(
            title=f"[{cat_names[cat]}] — {len(cat_results)} kernel(s)",
            box=box.ROUNDED,
            show_lines=True,
            width=200,
        )
        table.add_column("Kernel", style="bold", no_wrap=True, width=28)
        table.add_column("PP Stage", justify="center", width=8)
        table.add_column("Weight Shape (Orig→Shard)", width=24)
        table.add_column("Act Shape (Orig→Shard)", width=24)
        table.add_column("Shard Applied", width=32)
        table.add_column("Comm Fwd", width=30)
        table.add_column("Comm Bwd", width=24)
        table.add_column("Mem/GPU", justify="right", width=10)
        table.add_column("Notes", width=40)

        for r in cat_results:
            # Skip showing every layer if too many — show representative layers
            layer_idx = r.kernel.layer_idx
            if cfg.num_hidden_layers > 12 and layer_idx is not None:
                # Show layers 0, N//4, N//2, 3N//4, N-1 as representatives
                show_indices = {0, cfg.num_hidden_layers // 4, cfg.num_hidden_layers // 2, 
                                3 * cfg.num_hidden_layers // 4, cfg.num_hidden_layers - 1}
                if layer_idx not in show_indices:
                    continue

            pp_str = str(r.pp_stage) if r.pp_stage >= 0 else "global"
            if r.pp_stage >= 0 and pc.pp > 1:
                pp_str = f"[yellow]{r.pp_stage}[/yellow]"

            shape_str = _fmt_shape_diff(r.original_weight_shape, r.sharded_weight_shape)
            act_str = _fmt_shape_diff(r.original_act_shape, r.sharded_act_shape)
            mem_str = _fmt_bytes(r.memory_per_gpu_bytes) if r.memory_per_gpu_bytes > 0 else "—"

            table.add_row(
                r.kernel.name,
                pp_str,
                shape_str,
                act_str,
                r.shard_dims_applied,
                r.comm_fwd,
                r.comm_bwd,
                mem_str,
                r.notes,
            )

        if cat == "attention" and cfg.num_hidden_layers > 12:
            console.print()
            console.print(f"[dim italic]Showing 5 representative layers out of {cfg.num_hidden_layers}[/dim italic]")
        
        console.print(table)

    # ── Memory summary ──
    console.print()
    total_mem = sum(r.memory_per_gpu_bytes for r in results)
    total_orig = sum(r.total_memory_bytes for r in results)
    
    mem_table = Table(title="Memory Summary", box=box.ROUNDED, show_lines=True)
    mem_table.add_column("Metric", style="bold cyan")
    mem_table.add_column("Value", justify="right")
    mem_table.add_row("Total weight memory (full model)", _fmt_bytes(total_orig))
    mem_table.add_row("Per-GPU weight memory (sharded)", _fmt_bytes(total_mem))
    mem_table.add_row("Shard efficiency", f"{total_mem/max(total_orig,1)*100/pc.world_size*pc.dp:.1f}% of single-GPU (×{pc.world_size} GPUs)")
    console.print(mem_table)

    # ── Communication summary ──
    console.print()
    comm_ops = {}
    for r in results:
        for op in [r.comm_fwd, r.comm_bwd]:
            if op and op != "none":
                comm_ops[op] = comm_ops.get(op, 0) + 1
    
    if comm_ops:
        comm_table = Table(title="Communication Operations Summary", box=box.ROUNDED, show_lines=True)
        comm_table.add_column("Operation", style="bold yellow")
        comm_table.add_column("Count", justify="right")
        comm_table.add_column("Frequency", justify="right")
        for op, count in sorted(comm_ops.items(), key=lambda x: -x[1]):
            comm_table.add_row(op, str(count), f"{count}× per layer" if "layer" in op.lower() or count > 1 else "1×")
        console.print(comm_table)

    return console.export_text()


def render_kernel_table_plain(
    cfg: ModelConfig,
    pc: ParallelConfig,
    results: list[ShardedKernel],
) -> str:
    """Fallback plain-text table renderer."""
    lines = []
    lines.append("=" * 200)
    lines.append("Parallel Tool — Kernel Sharding Analysis")
    lines.append("=" * 200)
    lines.append(f"Model: {cfg.model_type} | {cfg.architectures}")
    lines.append(f"  Hidden={cfg.hidden_size}  Layers={cfg.num_hidden_layers}  "
                 f"Heads={cfg.num_attention_heads}  KV={cfg.kv_heads}  "
                 f"HeadDim={cfg.effective_head_dim}")
    lines.append(f"  Vocab={cfg.vocab_size}  Inter={cfg.intermediate_size}  "
                 f"MoE={cfg.is_moe}" +
                 (f"  Experts={cfg.num_experts}" if cfg.is_moe else ""))
    lines.append(f"Parallel: {pc}")
    lines.append(f"World Size: {pc.world_size} GPU(s)")
    lines.append("")

    # PP layout
    if pc.pp > 1:
        lines.append("-" * 80)
        lines.append("Pipeline Parallel — Stage Layout")
        lines.append("-" * 80)
        layers_per_stage = cfg.num_hidden_layers // pc.pp
        extra = cfg.num_hidden_layers % pc.pp
        cumulative = 0
        for s in range(pc.pp):
            stage_layers = layers_per_stage + (1 if s < extra else 0)
            end = cumulative + stage_layers - 1
            lines.append(f"  Stage {s}: layers [{cumulative} → {end}]  ({stage_layers} layers)")
            cumulative += stage_layers
        lines.append("")

    # Per-category
    categories = ["embedding", "attention", "mlp", "moe", "norm", "lm_head"]
    cat_names = {
        "embedding": "Embedding",
        "attention": "Attention (Q/KV/O + Core)",
        "mlp": "MLP / FFN (Gate/Up/Down)",
        "moe": "MoE (Router + Experts)",
        "norm": "LayerNorm / RMSNorm",
        "lm_head": "LM Head",
    }

    for cat in categories:
        cat_results = [r for r in results if r.kernel.category == cat]
        if not cat_results:
            continue

        lines.append("-" * 200)
        lines.append(f"[{cat_names[cat]}] — {len(cat_results)} kernel(s)")
        lines.append("-" * 200)
        
        # Header
        cols = ["Kernel", "PP", "Weight(Orig→Shard)", "Act(Orig→Shard)", 
                "Shard Applied", "Comm Fwd", "Mem/GPU"]
        lines.append(f"  {cols[0]:<30} {cols[1]:<6} {cols[2]:<26} {cols[3]:<26} {cols[4]:<36} {cols[5]:<36} {cols[6]:<12}")
        lines.append("  " + "-" * 196)

        for r in cat_results:
            layer_idx = r.kernel.layer_idx
            if cfg.num_hidden_layers > 12 and layer_idx is not None:
                show_indices = {0, cfg.num_hidden_layers // 4, cfg.num_hidden_layers // 2,
                                3 * cfg.num_hidden_layers // 4, cfg.num_hidden_layers - 1}
                if layer_idx not in show_indices:
                    continue

            orig_s = _fmt_shape(r.original_weight_shape)
            shard_s = _fmt_shape(r.sharded_weight_shape)
            shape_str = f"{orig_s} → {shard_s}" if orig_s != shard_s else orig_s

            orig_a = _fmt_shape(r.original_act_shape)
            shard_a = _fmt_shape(r.sharded_act_shape)
            act_str = f"{orig_a} → {shard_a}" if orig_a != shard_a else orig_a

            mem_str = _fmt_bytes(r.memory_per_gpu_bytes) if r.memory_per_gpu_bytes > 0 else "—"
            pp_str = str(r.pp_stage) if r.pp_stage >= 0 else "glb"

            lines.append(
                f"  {r.kernel.name:<30} {pp_str:<6} {shape_str:<26} {act_str:<26} "
                f"{r.shard_dims_applied:<36} {r.comm_fwd:<36} {mem_str:<12}"
            )
            if r.notes:
                lines.append(f"  {'':>30} {'':>6} {'':>26} {'':>26} {'':>36} {'':>36} └─ {r.notes}")

        lines.append("")

    # Memory summary
    lines.append("=" * 80)
    lines.append("Memory Summary")
    lines.append("=" * 80)
    total_mem = sum(r.memory_per_gpu_bytes for r in results)
    total_orig = sum(r.total_memory_bytes for r in results)
    lines.append(f"  Total weight memory (full model): {total_orig} bytes = {_fmt_bytes(total_orig)}")
    lines.append(f"  Per-GPU weight memory (sharded):  {total_mem} bytes = {_fmt_bytes(total_mem)}")
    lines.append("")

    # Communication summary
    comm_ops = {}
    for r in results:
        for op in [r.comm_fwd, r.comm_bwd]:
            if op and op != "none":
                comm_ops[op] = comm_ops.get(op, 0) + 1
    
    if comm_ops:
        lines.append("=" * 80)
        lines.append("Communication Operations Summary")
        lines.append("=" * 80)
        for op, count in sorted(comm_ops.items(), key=lambda x: -x[1]):
            lines.append(f"  {op:<50} ×{count}")
        lines.append("")

    return "\n".join(lines)


def render_table(
    cfg: ModelConfig,
    pc: ParallelConfig,
    results: list[ShardedKernel],
    use_rich: bool = True,
) -> str:
    """Render the full analysis table."""
    if use_rich and RICH_AVAILABLE:
        return render_kernel_table_rich(cfg, pc, results)
    else:
        return render_kernel_table_plain(cfg, pc, results)
