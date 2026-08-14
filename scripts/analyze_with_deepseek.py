#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transcribe an image, then ask DeepSeek to reason over the transcript."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def transcribe(image_path, extra_args):
    script = Path(__file__).with_name("img2txt.py")
    cmd = [sys.executable, str(script), str(image_path), *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit("img2txt failed:\n" + proc.stderr)
    return proc.stdout


def main():
    ap = argparse.ArgumentParser(description="Transcribe an image and analyze it with DeepSeek.")
    ap.add_argument("image")
    ap.add_argument("--ascii", type=int, default=48)
    ap.add_argument("--palette", type=int, default=6)
    ap.add_argument("--grid", default="6x4")
    ap.add_argument("--edge", type=int, default=0)
    ap.add_argument("--tiles", default="0x0")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the transcript only")
    ap.add_argument("--prompt", default="请分析这张图片：主要内容、布局、颜色和明暗。")
    args = ap.parse_args()

    extra = [
        "--ascii", str(args.ascii),
        "--palette", str(args.palette),
        "--grid", args.grid,
        "--edge", str(args.edge),
        "--tiles", args.tiles,
    ]
    if args.ocr:
        extra.append("--ocr")

    transcript = transcribe(args.image, extra)
    if args.dry_run:
        print(transcript)
        return

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("缺少 openai 依赖，请运行：pip install openai")

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    system = (
        "你是转录视觉推理器。你只能看到图片的结构化文本转录，不能看到图片本身。"
        "只依据转录内容推理，禁止脑补。先给观察（注明来源，如 ASCII 第 N 行 / "
        "COLOR GRID 坐标 / PALETTE 色值），再给推断和置信度；"
        "信息不足时输出 NEED_MORE: <建议参数>。"
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": transcript + "\n\n" + args.prompt},
        ],
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
