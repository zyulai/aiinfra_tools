"""Terminal-width-aware analysis views and side-effect-free text rendering."""
import io
import shutil
import textwrap

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .config_parser import ModelConfig
from .parallel_engine import ParallelConfig, ShardedKernel, pipeline_layer_ranges
from .reporting import (
    CATEGORY_NAMES, analyze_model, format_bytes, select_results, summarize_memory,
)

_fmt_bytes = format_bytes


def _terminal_width(width=None):
    return max(20, width or shutil.get_terminal_size((100, 24)).columns)


def _fmt_shape(shape: tuple) -> str:
    return "×".join(str(dim) for dim in shape) if shape and shape != (0,) else "—"


def _fmt_shape_diff(original: tuple, sharded: tuple) -> str:
    if original == sharded:
        return _fmt_shape(original)
    return f"{_fmt_shape(original)} → {_fmt_shape(sharded)}"


def _kernel_label(result: ShardedKernel) -> str:
    layer = result.kernel.layer_idx
    prefix = f"layer_{layer}_"
    if layer is not None and result.kernel.name.startswith(prefix):
        name = result.kernel.name[len(prefix):]
        if result.kernel.category == "moe" and name.startswith("moe_"):
            name = name[len("moe_"):]
        return f"L{layer} {name}"
    return result.kernel.name


def _shape_text(result):
    text = Text(_fmt_shape(result.original_weight_shape))
    if result.is_sharded:
        text.append(" → ", style="dim")
        for index, dim in enumerate(result.sharded_weight_shape):
            if index:
                text.append("×", style="dim")
            changed = dim != result.original_weight_shape[index]
            text.append(str(dim), style="bold green" if changed else None)
    return text


def _summary_lines(cfg, pc, summary):
    return [
        f"Model: {cfg.model_type} | Layers: {cfg.num_hidden_layers} | GPUs: {pc.world_size}",
        f"Parallel: {pc}",
        f"Full-model weights: {format_bytes(summary.full_weight_bytes)}",
        f"Peak weight/GPU: {format_bytes(summary.peak_weight_bytes)}",
        "Weights only; excludes activations, KV cache, gradients and workspace.",
        "Shapes: linear [in,out], embedding [vocab,hidden], norm/bias [hidden/out].",
    ]


def _stage_rows(cfg, pc, results, summary):
    ranges = pipeline_layer_ranges(cfg.num_hidden_layers, pc.pp)
    global_modules = [[] for _ in range(pc.pp)]
    for result in results:
        if result.kernel.layer_idx is None:
            global_modules[result.pp_stage].append(result.kernel.name)
    rows = []
    for stage, (start, end) in enumerate(ranges):
        globals_on_stage = global_modules[stage]
        rows.append((
            str(stage), f"{start}–{end - 1}", str(end - start),
            format_bytes(summary.stage_weight_bytes[stage]),
            ", ".join(globals_on_stage) or "—",
        ))
    return rows


def _detail_lines(result):
    lines = [
        f"HF tensor: {result.kernel.weight_name}",
        f"Activation/token: {_fmt_shape_diff(result.original_act_shape, result.sharded_act_shape)}",
        f"Forward: {result.comm_fwd}", f"Backward: {result.comm_bwd}",
    ]
    if result.notes:
        lines.append(f"Notes: {result.notes}")
    return lines


def _analysis_view(cfg, pc, results, *, width, detailed=False, **filters):
    selected = select_results(cfg, pc, results, **filters)
    summary = summarize_memory(cfg, pc, results)
    items = [Panel(Text("\n".join(_summary_lines(cfg, pc, summary))),
                   title="Kernel Sharding", border_style="blue")]
    if pc.pp > 1:
        table = Table(title="Pipeline stages · weight per GPU", box=box.ROUNDED, width=width)
        for name in ("Stage", "Layers", "Count", "Weight/GPU", "Global modules"):
            table.add_column(name, overflow="fold")
        for row in _stage_rows(cfg, pc, results, summary):
            table.add_row(*(Text(cell) for cell in row))
        items.append(table)
    items.append(Text(
        f"Showing {len(selected)} of {len(results)} kernels. "
        "Use layer/stage/category filters or show all layers to inspect more.", style="dim",
    ))
    if not selected:
        items.append(Text("No kernels match these filters.", style="yellow"))
    for category, title in CATEGORY_NAMES.items():
        rows = [result for result in selected if result.kernel.category == category]
        if not rows:
            continue
        if width < 64:
            for result in rows:
                lines = [
                    f"Stage {result.pp_stage} · {format_bytes(result.memory_per_gpu_bytes)} weight/GPU",
                    f"Shape: {_fmt_shape_diff(result.original_weight_shape, result.sharded_weight_shape)}",
                    f"Split: {result.shard_dims_applied}",
                ]
                items.append(Panel(Text("\n".join(lines)), title=_kernel_label(result)))
        else:
            table = Table(title=title, box=box.ROUNDED, width=width, show_lines=True)
            table.add_column("Kernel", ratio=2, overflow="fold")
            if pc.pp > 1:
                table.add_column("Stage", justify="center", width=5)
            table.add_column("Weight shape · original → per GPU", ratio=3, overflow="fold")
            table.add_column("Split", ratio=2, overflow="fold")
            table.add_column("Weight/GPU", justify="right", width=10)
            for result in rows:
                cells = [Text(_kernel_label(result), style="bold")]
                if pc.pp > 1:
                    cells.append(str(result.pp_stage))
                # Stage has its own column; keep split descriptions compact.
                split = "; ".join(part for part in result.shard_dims_applied.split("; ")
                                  if not part.startswith("PP:"))
                split = split.replace("replicated (no weight split)", "replicated")
                if split in ("", "none"):
                    split = "replicated" if result.memory_per_gpu_bytes else "no weight"
                cells.extend((_shape_text(result), Text(split), format_bytes(result.memory_per_gpu_bytes)))
                table.add_row(*cells)
            items.append(table)
        if category == "moe":
            items.append(Text(
                f"Expert shapes are per expert; weight bytes include all {cfg.num_experts // pc.ep} "
                "experts held by each rank.", style="dim",
            ))
        if detailed:
            for result in rows:
                items.append(Panel(Text("\n".join(_detail_lines(result))),
                                   title=_kernel_label(result), border_style="dim"))
    return Group(*items)


def render_kernel_table_rich(cfg, pc, results, *, width=None, detailed=False, **filters) -> str:
    """Render into a buffer; callers decide when and where to print."""
    width = _terminal_width(width)
    return _render_rich(_analysis_view(cfg, pc, results, width=width,
                                      detailed=detailed, **filters), width)


def _render_rich(view, width):
    buffer = io.StringIO()
    Console(file=buffer, width=width, color_system=None).print(view)
    return buffer.getvalue()


def render_kernel_table_plain(cfg, pc, results, *, width=None, detailed=False, **filters) -> str:
    width = _terminal_width(width)
    summary = summarize_memory(cfg, pc, results)
    lines = ["Kernel Sharding", *_summary_lines(cfg, pc, summary)]
    if pc.pp > 1:
        lines.append("\nPipeline stages:")
        for stage, layers, count, memory, globals_ in _stage_rows(cfg, pc, results, summary):
            lines.append(f"Stage {stage}: layers {layers} ({count}), {memory}/GPU; {globals_}")
    selected = select_results(cfg, pc, results, **filters)
    lines.append(f"\nShowing {len(selected)} of {len(results)} kernels")
    for result in selected:
        lines.extend((
            f"\n{_kernel_label(result)} | stage {result.pp_stage} | {format_bytes(result.memory_per_gpu_bytes)} weight/GPU",
            f"  Weight: {_fmt_shape_diff(result.original_weight_shape, result.sharded_weight_shape)}",
            f"  Split: {result.shard_dims_applied}",
        ))
        if detailed:
            lines.extend(f"  {line}" for line in _detail_lines(result))
    if not selected:
        lines.append("No kernels match these filters.")
    return "\n".join(
        textwrap.fill(line, width=width, subsequent_indent="  ", replace_whitespace=False)
        if line else "" for line in lines
    )


def render_table(cfg, pc, results, use_rich=True, **options) -> str:
    renderer = render_kernel_table_rich if use_rich and RICH_AVAILABLE else render_kernel_table_plain
    return renderer(cfg, pc, results, **options)


def print_analysis(cfg, pc, results, *, console=None, use_rich=True, width=None, **options):
    """Print exactly once, preserving Rich colors for interactive terminals."""
    if use_rich and RICH_AVAILABLE:
        console = console or Console(width=width)
        console.print(_analysis_view(cfg, pc, results, width=console.width, **options))
    else:
        print(render_kernel_table_plain(cfg, pc, results, width=width, **options))


def _comparison_view(cfg, strategies, analyses, *, width):
    table = Table(title="Strategy comparison · weights only", box=box.ROUNDED, width=width)
    for heading in ("Plan", "Configuration", "GPUs", "Peak weight/GPU", "Stage weights"):
        table.add_column(heading, overflow="fold")
    for index, (pc, results) in enumerate(zip(strategies, analyses), 1):
        summary = summarize_memory(cfg, pc, results)
        table.add_row(str(index), Text(str(pc)), str(pc.world_size), format_bytes(summary.peak_weight_bytes),
                      Text(", ".join(format_bytes(value) for value in summary.stage_weight_bytes)))
    items = [table, Text("Linear shapes use [in,out]. Expert shapes are per expert.", style="dim")]
    names = set()
    for pc, results in zip(strategies, analyses):
        names.update(result.kernel.name for result in select_results(cfg, pc, results))
    batch_size = 2 if width < 120 else 3
    for start in range(0, len(strategies), batch_size):
        shapes = Table(title="Per-kernel weight shapes", box=box.ROUNDED, width=width)
        shapes.add_column("Kernel", ratio=2, overflow="fold")
        batch = list(zip(strategies[start:start + batch_size], analyses[start:start + batch_size]))
        for index in range(start, start + len(batch)):
            shapes.add_column(f"Plan {index + 1}", ratio=2, overflow="fold")
        for row_index, reference in enumerate(analyses[0]):
            if reference.kernel.name not in names:
                continue
            cells = [Text(_kernel_label(reference))]
            for pc, results in batch:
                result = results[row_index]
                shape = _fmt_shape(result.sharded_weight_shape)
                if result.kernel.shard_type == "expert":
                    shape += f" × {cfg.num_experts // pc.ep} experts"
                cells.append(Text(shape))
            shapes.add_row(*cells)
        items.append(shapes)
    return Group(*items)


def _comparison_analyses(cfg, strategies, dtype):
    if not strategies:
        raise ValueError("Select at least one strategy")
    # One complete analysis per plan, reused for summaries and every table row.
    return [analyze_model(cfg, pc, dtype) for pc in strategies]


def render_comparison(cfg, strategies, *, width=None, dtype="bfloat16", use_rich=True) -> str:
    analyses = _comparison_analyses(cfg, strategies, dtype)
    width = _terminal_width(width)
    if use_rich and RICH_AVAILABLE:
        return _render_rich(_comparison_view(cfg, strategies, analyses, width=width), width)
    lines = ["Strategy comparison · weights only"]
    for index, (pc, results) in enumerate(zip(strategies, analyses), 1):
        lines.append(f"\nPlan {index}: {pc} | GPUs={pc.world_size}")
        lines.append(render_kernel_table_plain(cfg, pc, results, width=width))
    return "\n".join(lines)


def print_comparison(cfg, strategies, *, console=None, dtype="bfloat16", width=None, use_rich=True):
    if use_rich and RICH_AVAILABLE:
        console = console or Console(width=width)
        analyses = _comparison_analyses(cfg, strategies, dtype)
        console.print(_comparison_view(cfg, strategies, analyses, width=console.width))
    else:
        print(render_comparison(cfg, strategies, dtype=dtype, width=width, use_rich=False))
