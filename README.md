# text-model-vision

让纯文本大模型（以 DeepSeek 为标准）获得图像视觉能力，全程不调用 GLM 或其他视觉/多模态 API。核心思路是把图片先本地转成结构化文本转录，再让纯文本模型基于转录进行推理。
## 自动安装提示词：

```text
帮我安装这个skill：https://github.com/hyzinc/Eyes-for-deepseek-ect
```
安装后直接说：

```text
使用 text-model-vision 分析这张图片：<图片路径>
```

## 特性

- 不依赖任何多模态 API，无 GLM-4V、GPT-4V、Claude vision。
- 只依赖 Pillow，本地确定性转录，CPU 即可运行。
- 输出 ASCII 字符画、色块网格、主色调、边缘图、分区裁剪和可选本地 OCR。
- 为 DeepSeek 内置“只依据转录证据推理”的提示词协议，降低幻觉。
- 支持 `--crop` 与 TILES 多轮放大，信息不足时逐级细化。
- 可直接作为 Codex Skill 安装使用。

## 工作原理

```text
图片文件 -> scripts/img2txt.py -> 转录文本 -> DeepSeek -> 结论
           (本地、确定性)                    (纯文本推理)
```

纯文本模型无法直接读取像素，因此先由本地脚本把图片编码成结构化文本，再让模型像阅读证据一样推理。答案必须能在转录中找到依据。

## 目录结构

```text
text-model-vision/
├── SKILL.md                      # Codex Skill 定义（含完整提示词协议）
├── README.md
├── requirements.txt              # Python 依赖
├── environment.yml               # conda 环境配置
├── .env.example                  # DeepSeek API 环境变量示例
├── agents/
│   └── openai.yaml               # Codex UI 元数据
├── scripts/
│   ├── img2txt.py                # 图片 -> 文本转录 CLI
│   └── analyze_with_deepseek.py  # 转录 + DeepSeek API 一键分析
├── examples/
│   ├── demo-image.png
│   └── demo-transcript.txt
├── tests/
│   └── test_img2txt.py
└── .github/workflows/ci.yml
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或使用 conda：

```bash
conda env create -f environment.yml
conda activate text-model-vision
```

### 2. 生成转录

```bash
python scripts/img2txt.py examples/demo-image.png
python scripts/img2txt.py examples/demo-image.png --ascii 64 --edge 48 --tiles 3x3
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--ascii 48` | ASCII 字符画宽度，0 关闭 |
| `--edge 40` | 边缘图宽度，0 关闭 |
| `--palette 6` | 主色数量，0 关闭 |
| `--grid 6x4` | 色块网格 宽x高，0 关闭 |
| `--tiles 3x3` | 分区裁剪 行x列，0 关闭 |
| `--crop x0,y0,x1,y1` | 先裁剪再转录 |
| `--ocr` | 启用本地 tesseract（可选） |

### 3. 交给 DeepSeek

把转录文本原样粘贴进 DeepSeek，并附加：

```text
以下是图片转录文本。只依据转录内容分析图片，不得补充转录之外的信息。
先给观察（注明来源，如 ASCII 第 N 行 / COLOR GRID 坐标 / PALETTE 色值），
再给推断和置信度；信息不足时输出 NEED_MORE。
```

### 4. 一键分析（DeepSeek API）

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
python scripts/analyze_with_deepseek.py examples/demo-image.png
```

只查看转录不调用 API：

```bash
python scripts/analyze_with_deepseek.py examples/demo-image.png --dry-run
```

## 作为 Codex Skill 安装

把整个仓库放到 `~/.codex/skills/text-model-vision/`，或在 Codex 中引用本仓库路径：

```text
~/.codex/skills/text-model-vision/
├── SKILL.md
└── scripts/img2txt.py
```



## 测试

```bash
python -m unittest discover -s tests
```

## 限制

- 转录是低分辨率量化，细小文字、人脸细节、复杂纹理不可靠。
- 模型只能依据转录推理，必须禁止脑补。
- 文字提取需要可选安装本地 tesseract。
- 本技能提供“基于结构化证据的文本推理视觉”，不是真实视觉。

## License

MIT
