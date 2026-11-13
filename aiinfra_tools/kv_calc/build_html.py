#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 presets.json 同步进 kv_cache_calculator.html 的 `const PRESETS = [...]` 一行。
用法:先跑 build_presets.py 更新预设,再跑本脚本。
HTML 其余部分(布局、JS 公式)直接在 kv_cache_calculator.html 里维护。
"""
import argparse
import json
import os
import re

SHELL = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV Cache 显存计算器</title>
<style>
:root {
  --background: #f7f7f8; --foreground: #18181b; --card: #ffffff;
  --card-foreground: #18181b; --muted: #ececee; --muted-foreground: #6b6b74;
  --border: #e2e2e6; --primary: #27272a; --primary-foreground: #fafafa;
  --viz-series-1: #3b82f6; --viz-series-2: #f97316; --viz-series-3: #10b981;
  --viz-series-4: #a855f7;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --background: #131316; --foreground: #f2f2f4; --card: #1c1c20;
    --card-foreground: #f2f2f4; --muted: #26262c; --muted-foreground: #9a9aa4;
    --border: #2c2c33; --primary: #f2f2f4; --primary-foreground: #18181b;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 20px; background: var(--background); color: var(--foreground);
  font: 400 14px/1.55 system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; }
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 24px; font-weight: 500; margin: 0 0 2px; }
h3 { font-size: 16px; font-weight: 500; margin: 18px 0 6px; }
p { margin: 6px 0; }
code { font-family: ui-monospace, Consolas, monospace; background: var(--muted);
  padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.text-small { font-size: 12px; }
.text-muted { color: var(--muted-foreground); }
.tabular-nums { font-variant-numeric: tabular-nums; }
.card { background: var(--card); color: var(--card-foreground); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; }
.viz-stat .viz-stat-value { font-size: 20px; font-weight: 500; margin: 2px 0; }
.viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.viz-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.viz-controls { display: flex; flex-wrap: wrap; gap: 14px 18px; align-items: flex-end; }
.viz-controls > div { min-width: 130px; }
.form-label { display: block; font-size: 12px; color: var(--muted-foreground); margin-bottom: 3px; }
.form-select, .form-control { height: 30px; padding: 2px 8px; border: 1px solid var(--border);
  border-radius: 7px; background: var(--card); color: var(--foreground); font-size: 13px; }
.form-select-sm { height: 28px; }
.form-control-sm { height: 28px; width: 110px; }
.form-range { width: 150px; accent-color: var(--primary); }
.table { width: 100%; border-collapse: collapse; font-size: 13px; }
.table th { text-align: left; color: var(--muted-foreground); font-weight: 500;
  padding: 5px 8px; border-bottom: 1px solid var(--border); }
.table td { padding: 5px 8px; border-bottom: 1px solid var(--border); }
.table-responsive { overflow-x: auto; }
.text-end { text-align: right; }
</style>
</head>
<body>
<main>
<h1>KV Cache 显存计算器</h1>
<p class="text-small text-muted">离线独立版 · 计算口径与 vLLM 最新 main 分支一致 ·
配套命令行版:<code>python kv_calculator.py &lt;模型ID&gt;</code></p>
<!--__FRAGMENT__-->
</main>
</body>
</html>
"""


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=os.path.join(here, "kv_cache_calculator.html"),
                    help="要就地更新的 HTML 计算器")
    ap.add_argument("--presets", default=os.path.join(here, "presets.json"),
                    help="build_presets.py 生成的预设库")
    args = ap.parse_args()
    presets = json.load(open(args.presets, encoding="utf-8"))
    src = open(args.html, encoding="utf-8").read()
    line = "const PRESETS = " + json.dumps(presets, ensure_ascii=False) + ";"
    new_src, n = re.subn(r"^const PRESETS = .*;$", lambda m: line,
                         src, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{args.html} 里没有找到 `const PRESETS = ...;` 行")
    open(args.html, "w", encoding="utf-8").write(new_src)
    print(f"{args.html}: 已同步 {len(presets)} 个预设 "
          f"({os.path.getsize(args.html)//1024} KB)")


if __name__ == "__main__":
    main()
