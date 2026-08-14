---
name: text-model-vision
description: 让纯文本大模型（以 DeepSeek 为标准）获得图像视觉能力，全程不调用 GLM 等视觉或多模态 API。使用本地确定性脚本把图片转成 ASCII 字符画、色块网格、主色调、边缘图、分区裁剪等文本转录，再让纯文本模型基于转录进行推理。当用户提供图片并要求描述、分析、计数、读图表、比较图像、识别布局或提取画面信息，而可用模型只有纯文本能力时使用本技能。
---

# 纯文本模型视觉技能（DeepSeek 标准）

本技能的目标：给只支持文本输入输出的 AI 模型（以 DeepSeek 为标准）提供可用的图像视觉能力，不依赖 GLM 或其他多模态模型，只依靠本地确定性计算和模型自身的文本推理能力。

## 1. 核心原理

纯文本模型无法直接读取像素，因此把“视觉”拆成两步：

1. **转录**：本地脚本 `img2txt.py` 把图片编码成结构化文本，包括亮度字符画、色块网格、主色调、边缘轮廓和分区裁剪。
2. **推理**：DeepSeek 只基于这份转录文本进行观察、定位、计数和结论，不接触图片本身。

约束：全程不调用 GLM-4V、GPT-4V、Claude vision 等任何视觉 API；模型看到的全部内容来自转录文本，因此答案必须在转录中能找到证据。

工作流：

```text
图片文件 -> img2txt.py -> 转录文本 -> DeepSeek 推理 -> 结论
           (本地、确定性)              (纯文本、自身算力)
```

## 2. 能力与边界

可以做到：

- 描述画面构图、主要元素、颜色、明暗和布局。
- 估计目标数量、位置和大小比例。
- 读取柱状图、折线图、饼图等图表趋势。
- 划分 UI、网页、报表的区域结构。
- 比较两张图片的异同。
- 通过 `--crop` 或 TILES 放大局部后多轮推理。
- 可选调用本地 tesseract 提取文字（纯本地工具，非视觉模型）。

边界：

- 转录是低分辨率量化，细小文字、人脸细节、复杂纹理不可靠。
- 禁止脑补转录中不存在的细节；模糊时必须输出“转录中不可见”或请求放大。
- 字符画对低对比度、大面积单色图不敏感，此时优先使用 COLOR GRID 和 PALETTE。
- 本技能提供的是“基于结构化证据的文本推理视觉”，不是真实视觉。

## 3. 快速开始

安装依赖：

```bash
pip install pillow
```

可选 OCR（本地工具）：

```bash
# Windows
choco install tesseract
# Ubuntu
sudo apt-get install tesseract-ocr
```

生成转录：

```bash
python img2txt.py photo.png
python img2txt.py photo.png --ascii 64 --edge 48 --tiles 3x3
python img2txt.py screenshot.png --ocr --tiles 3x3 --tile-ascii 20
python img2txt.py photo.png --crop 320,0,480,180 --ascii 40
```

然后把转录文本原样粘贴进 DeepSeek，并附加第 9 节的提示词模板。

## 4. 编码模式

| 模式 | 输出内容 | 适用场景 |
| --- | --- | --- |
| META | 尺寸、宽高比、亮度、对比度、整体主色 | 一切任务的基线 |
| PALETTE | 前 N 个主色及面积占比 | 配色、图表系列、场景风格 |
| COLOR GRID | WxH 的平均色网格 | 区域布局、粗略构图 |
| ASCII | 亮度字符画，宽高比已校正 | 形状、位置、大小、明暗分布 |
| EDGE MAP | 轮廓与边界 | 元素边界、图表折线、区域分隔 |
| TILES | 分区裁剪 + 每块小字符画 | 局部放大、第二轮推理 |
| OCR | 本地 tesseract 提取文字 | 截图、文档、图表标签 |

命令行参数：

```text
--ascii 48         ASCII 宽度（字符数），0 关闭
--edge 40          边缘图宽度（字符数），0 关闭
--palette 6        主色数量，0 关闭
--grid 6x4         色块网格 宽x高，0 关闭
--tiles 3x3        分区裁剪 行x列，0 关闭
--tile-ascii 16    每个分区的小字符画宽度
--contrast 1.2     对比度增强倍数
--crop x0,y0,x1,y1 先裁剪再转录
--ocr              启用本地 OCR
```

## 5. Token 预算

| 档位 | 推荐参数 | 转录规模 | 大致 tokens |
| --- | --- | --- | --- |
| 快速 | `--ascii 32 --grid 4x3 --palette 4` | 约 0.4k 字符 | 约 400 |
| 默认 | `--ascii 48 --grid 6x4 --palette 6` | 约 0.9k 字符 | 约 900 |
| 详细 | `--ascii 72 --edge 56 --grid 8x6 --palette 8` | 约 1.7k 字符 | 约 1800 |
| 局部 | `--tiles 3x3 --tile-ascii 20` | 每轮约 1.2k 字符 | 约 1200 |

策略：先默认档，信息不足再对候选区域做 `--crop` 或 TILES 放大，避免一次性把上下文撑爆。

## 6. 工具脚本 `img2txt.py`

将以下代码保存为 `img2txt.py`。脚本只使用 Pillow，不调用任何网络服务或多模态 API。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""img2txt.py - deterministic image-to-text transcript for text-only LLMs.

No multimodal API is used. The output is a plain-text transcript (ASCII art,
color palette, color grid, edges, tiles) that a text-only model such as
DeepSeek can reason over.
"""

import argparse
import math
import subprocess
import sys
import tempfile
import os
from collections import Counter

from PIL import Image, ImageFilter, ImageOps


RAMP = "@%#*+=-:. "  # dark -> light, 10 levels


def clamp255(v):
    return max(0, min(255, int(round(v))))


def hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    h = 0.0
    if d > 0:
        if mx == r:
            h = 60.0 * (((g - b) / d) % 6)
        elif mx == g:
            h = 60.0 * ((b - r) / d + 2)
        else:
            h = 60.0 * ((r - g) / d + 4)
    s = 0.0 if mx == 0 else d / mx
    v = mx
    return h, s, v


def color_name(r, g, b):
    h, s, v = hsv(r, g, b)
    if v < 0.12:
        return "black"
    if s < 0.12:
        if v > 0.85:
            return "white"
        if v > 0.45:
            return "light-gray"
        return "dark-gray"
    if 10 <= h < 50 and s > 0.35 and v < 0.55:
        return "brown"
    if v < 0.30:
        return "dark-" + _hue_name(h, s)
    if v > 0.88 and s < 0.35:
        return "pale-" + _hue_name(h, s)
    if s < 0.45 and v > 0.72:
        return "light-" + _hue_name(h, s)
    return _hue_name(h, s)


def _hue_name(h, s):
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "orange" if s > 0.55 else "yellow"
    if h < 65:
        return "yellow"
    if h < 150:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 285:
        return "purple"
    if h < 330:
        return "pink"
    return "red"


def hex_of(rgb):
    return "#%02x%02x%02x" % tuple(clamp255(x) for x in rgb)


def ascii_grid(img, cols, rows=None, invert=False, contrast=1.0):
    w, h = img.size
    if rows is None:
        rows = max(1, int(round(cols * (h / float(w)) * 0.5)))
    small = img.resize((cols, rows), Image.BILINEAR)
    gray = small.convert("L")
    if invert:
        gray = ImageOps.invert(gray)
    px = gray.load()
    lines = []
    for y in range(rows):
        line = []
        for x in range(cols):
            v = px[x, y]
            v = clamp255((v - 127.5) * contrast + 127.5)
            idx = v * (len(RAMP) - 1) // 255
            line.append(RAMP[idx])
        lines.append("".join(line))
    return lines, rows, cols


def cell_averages(img, cols, rows):
    small = img.resize((cols, rows), Image.BILINEAR)
    px = small.load()
    out = []
    for y in range(rows):
        row = []
        for x in range(cols):
            rgb = px[x, y][:3]
            row.append((hex_of(rgb), color_name(*rgb)))
        out.append(row)
    return out


def palette(img, n=8, sample=96):
    small = img.copy()
    small.thumbnail((sample, sample), Image.BILINEAR)
    px = small.convert("RGB").load()
    counter = Counter()
    w, h = small.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y][:3]
            bucket = (r // 16 * 16, g // 16 * 16, b // 16 * 16)
            counter[bucket] += 1
    total = float(sum(counter.values())) or 1.0
    items = []
    for (rgb, cnt) in counter.most_common(n):
        items.append((hex_of(rgb), color_name(*rgb), cnt / total))
    return items


def brightness_contrast(img):
    small = img.convert("L").resize((64, 64), Image.BILINEAR)
    px = small.load()
    vals = [px[x, y] for y in range(64) for x in range(64)]
    mean = sum(vals) / float(len(vals))
    var = sum((v - mean) ** 2 for v in vals) / float(len(vals))
    return mean / 255.0, math.sqrt(var) / 255.0


def tiles(img, rows=3, cols=3, ascii_cols=16):
    w, h = img.size
    out = []
    for ty in range(rows):
        for tx in range(cols):
            x0 = int(w * tx / cols)
            x1 = int(w * (tx + 1) / cols)
            y0 = int(h * ty / rows)
            y1 = int(h * (ty + 1) / rows)
            crop = img.crop((x0, y0, x1, y1))
            avg = crop.resize((1, 1), Image.BILINEAR).load()[0, 0][:3]
            lines, r_, c_ = ascii_grid(crop, ascii_cols)
            out.append(
                {
                    "id": "t%d%d" % (ty + 1, tx + 1),
                    "box": (x0, y0, x1, y1),
                    "hex": hex_of(avg),
                    "name": color_name(*avg),
                    "brightness": brightness_contrast(crop)[0],
                    "ascii": lines,
                }
            )
    return out


def run_ocr(img):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        img.convert("RGB").save(tmp, format="PNG")
        proc = subprocess.run(
            ["tesseract", tmp, "stdout"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = proc.stdout.strip()
        if not text:
            return "[OCR] no text detected"
        return "[OCR] detected text:\n" + text
    except FileNotFoundError:
        return "[OCR] tesseract not installed (optional; core modes do not need it)"
    except Exception as exc:
        return "[OCR] failed: %s" % exc
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def render_transcript(img, args):
    w, h = img.size
    bright, contrast = brightness_contrast(img)
    lines = []

    lines.append("## META")
    lines.append("- size: %dx%d" % (w, h))
    lines.append("- aspect: %.2f" % (w / float(max(1, h))))
    lines.append("- brightness: %.2f (0=black, 1=white)" % bright)
    lines.append("- contrast: %.2f (0=flat, 1=high)" % contrast)
    small = img.convert("RGB").resize((1, 1), Image.BILINEAR).load()[0, 0]
    lines.append("- overall_color: %s (%s)" % (hex_of(small), color_name(*small)))
    lines.append("")

    if args.palette:
        lines.append("## PALETTE top %d (by area ratio)" % args.palette)
        for hexv, name, ratio in palette(img, args.palette):
            lines.append("- %s %s %.1f%%" % (hexv, name, ratio * 100))
        lines.append("")

    if args.grid:
        gcols, grows = args.grid
        lines.append("## COLOR GRID %dx%d (left-to-right, top-to-bottom)" % (gcols, grows))
        for row in cell_averages(img, gcols, grows):
            lines.append(" | ".join("%s %s" % cell for cell in row))
        lines.append("")

    if args.ascii:
        art, rows, cols = ascii_grid(img, args.ascii, contrast=args.contrast)
        lines.append("## ASCII %dx%d (darker char = darker pixel; aspect corrected)" % (cols, rows))
        lines.extend(art)
        lines.append("")

    if args.edge:
        edge = img.convert("L").filter(ImageFilter.FIND_EDGES)
        art, rows, cols = ascii_grid(edge, args.edge, invert=True, contrast=args.contrast)
        lines.append("## EDGE MAP %dx%d (outlines only)" % (cols, rows))
        lines.extend(art)
        lines.append("")

    if args.tiles:
        trows, tcols = args.tiles
        lines.append("## TILES %dx%d (zoomed crops; coordinates are pixel boxes)" % (trows, tcols))
        for t in tiles(img, trows, tcols, args.tile_ascii):
            lines.append("### %s box=%s avg=%s (%s) brightness=%.2f" % (
                t["id"], t["box"], t["hex"], t["name"], t["brightness"]))
            lines.extend(t["ascii"])
            lines.append("")

    if args.ocr:
        lines.append(run_ocr(img))
        lines.append("")

    return "\n".join(lines)


def parse_dims(s, default=8):
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    return int(s), default


def main():
    ap = argparse.ArgumentParser(description="Convert an image into a text transcript for a text-only LLM.")
    ap.add_argument("image")
    ap.add_argument("--ascii", type=int, default=48, help="ASCII art width in chars (0 to disable)")
    ap.add_argument("--edge", type=int, default=0, help="edge map width in chars (0 to disable)")
    ap.add_argument("--palette", type=int, default=6, help="palette size (0 to disable)")
    ap.add_argument("--grid", default="6x4", help="color grid WxH (0 to disable)")
    ap.add_argument("--tiles", default="0x0", help="tile crops RxC, e.g. 3x3 (0 to disable)")
    ap.add_argument("--tile-ascii", type=int, default=16)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--ocr", action="store_true", help="optional local tesseract OCR")
    ap.add_argument("--crop", default="", help="crop box x0,y0,x1,y1 before transcribing")
    args = ap.parse_args()

    if args.grid == "0":
        args.grid = None
    else:
        args.grid = parse_dims(args.grid)
    if args.tiles == "0x0":
        args.tiles = None
    else:
        args.tiles = parse_dims(args.tiles)

    img = Image.open(args.image).convert("RGB")
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        img = img.crop((x0, y0, x1, y1))
    print(render_transcript(img, args))


if __name__ == "__main__":
    main()
```

## 7. 转录示例

对一张 480x360 的合成场景图（蓝天、黄色太阳、绿色山坡、红色矩形）运行：

```bash
python img2txt.py sample.png --ascii 56 --palette 8 --grid 8x6
```

转录开头：

```text
## META
- size: 480x360
- aspect: 1.33
- brightness: 0.58 (0=black, 1=white)
- contrast: 0.16 (0=flat, 1=high)
- overall_color: #7fa099 (cyan)

## PALETTE top 8 (by area ratio)
- #208020 green 27.4%
- #90b0d0 light-blue 21.5%
- #80c0e0 light-blue 21.1%
- #90a0d0 light-blue 10.9%
- #c02020 red 7.0%
- #f0d030 yellow 2.4%

## COLOR GRID 8x6 (left-to-right, top-to-bottom)
#8ac8e7 light-blue | #8ac8e7 light-blue | ... | #90c9de light-blue
...
#268922 green | #387e23 green | ... | #228b22 green

## ASCII 56x21 (darker char = darker pixel; aspect corrected)
--------------------------------------------------------
--------------------------------------------------------
-----------------------------------------:::::::--------
...
------=++++++++++++++=----------------------------------
------=**************=----------------------------------
...
```

DeepSeek 基于该转录应给出的典型回答（示例）：

```text
观察：PALETTE 显示绿色占 27.4%、浅蓝约占 56%，红色占 7.0%，黄色占 2.4%；ASCII 上部多为浅色 "-"，中部有 ":::::" 亮斑，左下到右下是大片深色 "+/*"。
推断：画面上方是大面积浅蓝色天空，右上有一个较小的黄色亮斑（推测为太阳），画面下半部被绿色区域占据（推测为山坡/草地），中部偏左有一个深色红色矩形区域。
置信度：中高。依据是 COLOR GRID 第 4 行中列出现 red 色块、PALETTE 中 red 7.0%，且 ASCII 中对应位置为连续深色块。
```

## 8. DeepSeek 推理协议（强制）

把下面规则放进系统提示词或任务提示词，DeepSeek 每次分析都必须遵守：

1. **证据优先**：只引用转录中出现的内容，标注来源，例如 `ASCII 第 10 行`、`COLOR GRID (3,2)`、`PALETTE #c02020`。
2. **禁止脑补**：转录中没有的细节写“转录中不可见”，绝不编造物体、文字、数量或颜色。
3. **坐标规则**：COLOR GRID 行从上到下、列从左到右；TILES 的 `t11` 是左上角，`t23` 是第 2 行第 3 列；`box=` 是真实像素坐标。
4. **计数方法**：先按连续暗色或同色块圈出候选区域，逐区计数，再说明排除项；输出“估计值 + 依据”，不是精确值。
5. **图表方法**：先读标题、坐标轴（有 OCR 时），再用 PALETTE 区分数据系列，最后用 ASCII/EDGE 比较柱高、折线走向或饼图比例。
6. **颜色描述**：以 PALETTE 和 COLOR GRID 的 hex 与色名为准，不要自行改色名。
7. **回答格式**：先给“观察（证据）”，再给“推断（置信度：高/中/低）”。
8. **信息不足**：输出 `NEED_MORE: <建议参数>`（例如 `--crop 320,0,480,180 --ascii 48` 或 `--tiles 3x3`），由调用方做第二轮转录，而不是猜答案。

## 9. 提示词模板

### 通用描述

```text
以下是图片转录文本。只依据转录内容分析图片，不得补充转录之外的信息。

转录：
<TRANSCRIPT>

请输出：
1. 观察：列出与任务相关的证据（注明来源，如 ASCII 第 N 行 / COLOR GRID 坐标 / PALETTE 色值）。
2. 推断：画面主要内容、布局、颜色、明暗、风格。
3. 置信度：高/中/低，并说明依据。
```

### 目标计数

```text
转录：
<TRANSCRIPT>

任务：统计画面中的 <目标> 数量。
方法：先根据 ASCII 和 COLOR GRID 圈出候选区域，逐区计数并交叉验证，最后给出估计范围和置信度。
禁止凭空补出转录中不存在的目标。
```

### 图表读取

```text
转录：
<TRANSCRIPT>

任务：读取图表。
步骤：
1. 用 OCR/转录识别标题和坐标轴标签（没有就说明）。
2. 用 PALETTE 区分数据系列。
3. 用 ASCII/EDGE MAP 比较柱状图高度、折线走向或扇区大小。
4. 输出趋势结论和大致数值范围。
```

### UI / 布局分析

```text
转录：
<TRANSCRIPT>

任务：描述界面布局。
输出：从上到下、从左到右列出区域，包括位置、近似尺寸、背景色和亮色元素；
使用 COLOR GRID 坐标描述区域位置，例如“COLOR GRID 第 1-2 行全为深色，疑似顶部导航栏”。
```

### 两张图对比

```text
图 A 转录：
<TRANSCRIPT_A>

图 B 转录：
<TRANSCRIPT_B>

任务：逐项对比两图：整体色调、布局、主要元素、明暗差异、疑似内容变化。
每项差异都必须给出两边的转录证据。
```

### 文字提取

```text
转录：
<TRANSCRIPT>

任务：提取转录中 OCR 检测到的文字。
规则：逐字保留，不纠错、不补全、不翻译；OCR 未给出的文字写“无”。
```

### 多轮放大

```text
第一轮转录：
<TRANSCRIPT>

我怀疑目标位于 <区域描述>。请先说明你需要的精确裁剪坐标（box），
输出格式：CROP: x0,y0,x1,y1；我会用该坐标生成第二轮转录后再继续分析。
```

## 10. 多轮放大流程

1. 第一轮用默认档生成转录，DeepSeek 输出整体结构和候选区域。
2. 对候选区域执行：

```bash
python img2txt.py photo.png --crop 320,0,480,180 --ascii 48 --grid 6x4 --tiles 2x2
```

3. 把第二轮转录回传，DeepSeek 基于新证据细化结论。
4. 如果仍不清晰，继续缩小 `--crop` 或调高 `--ascii`；每轮控制在 1500 tokens 以内。

## 11. 部署接入

### DeepSeek API

```python
import subprocess
from openai import OpenAI

client = OpenAI(api_key="YOUR_DEEPSEEK_KEY", base_url="https://api.deepseek.com")

transcript = subprocess.run(
    ["python", "img2txt.py", "photo.png", "--ascii", "48"],
    capture_output=True, text=True, check=True
).stdout

resp = client.chat.completions.create(
    model="deepseek-chat",
    temperature=0.3,
    messages=[
        {"role": "system", "content": "你是转录视觉推理器。只能依据转录文本推理，禁止脑补。"},
        {"role": "user", "content": transcript + "\n\n请描述这张图片。"},
    ],
)
print(resp.choices[0].message.content)
```

### 本地 Ollama / vLLM

流程完全相同：先用 `img2txt.py` 离线生成转录，再把转录作为用户消息发给纯文本模型。可以把转录缓存成文件，供批量分析复用。

### 作为 Codex 技能安装

推荐目录结构：

```text
~/.codex/skills/text-model-vision/
├── SKILL.md              # 本文件
└── scripts/
    └── img2txt.py        # 第 6 节脚本
```

## 12. 验收与测试

每次修改或部署后执行：

1. 脚本自测：对一张已知图片运行默认命令，确认 META、PALETTE、COLOR GRID、ASCII 均非空且坐标正确。
2. 推理自测：让 DeepSeek 描述该图，逐条检查其结论能否在转录中找到证据。
3. 幻觉检查：人为在提示词中强调“只依据转录”，确认模型不会补出转录中不存在的物体。
4. 放大验证：用 `--crop` 对局部做第二轮，确认第二轮结论更准确。

## 13. 常见问题

**为什么不用 GLM？**
本技能不调用任何多模态 API。视觉能力由“本地确定性转录 + DeepSeek 自身文本推理”组成，GLM 只作为对比对象，不是依赖项。

**图片很大怎么办？**
脚本会把图片缩放到字符画和色块网格，转录体量只取决于网格参数，与原始像素数无关。先用默认档，再按需裁剪放大。

**能识别文字吗？**
核心管线不做文字识别；需要时可启用本地 tesseract（`--ocr`）。它仍是本地工具，不是多模态模型。

**能识别人脸或表情吗？**
只能给出颜色、明暗、轮廓和位置层面的推断，不能可靠识别身份或细微表情。

**需要 GPU 吗？**
不需要。转录脚本 CPU 即可在数秒内完成；推理部分取决于你部署 DeepSeek 的方式。
