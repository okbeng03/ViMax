# ComfyUI 工作流配置指南

本文档说明如何为 ViMax 配置 ComfyUI 工作流。

## 工作流文件结构

```
ViMax/
├── workflows/                    # 工作流文件夹
│   ├── flux_klein.json          # Flux 图片生成工作流
│   └── ltx2_3.json              # LTX 视频生成工作流
├── configs/
│   └── comfyui_example.yaml     # 配置文件模板
└── ...
```

## 1. Flux.klein 图片生成工作流

### 1.1 工作流要求

Flux.klein 工作流需要：
- 文本输入节点（text_g, text_l）
- 可选：参考图片输入节点（用于 IP-Adapter 等）

### 1.2 配置示例

```yaml
image_generator:
  class_path: tools.ImageGeneratorComfyUIFlux
  init_args:
    base_url: http://127.0.0.1:8188
    workflow_json_path: workflows/flux_klein.json
    input_node_descriptions:
      - node_id: "3"
        field: "text_g"
        source: prompt
      - node_id: "4"
        field: "text_l"
        source: prompt
      - node_id: "5"
        field: "image"
        source: reference_image
        max_count: 4
```

### 1.3 如何获取 node_id

1. 在 ComfyUI 中打开工作流
2. 右键点击节点 → "Copy node path" 或查看节点 ID
3. 节点 ID 通常是数字字符串（如 "3", "5", "10"）

## 2. LTX 2.3 视频生成工作流

### 2.1 工作流要求

LTX 2.3 工作流需要：
- 文本提示词节点（prompt）
- 多帧输入节点（image1, image2, ..., image6）
- 输出节点（VHS_VideoCombine 等）

### 2.2 配置示例

```yaml
video_generator:
  class_path: tools.VideoGeneratorComfyUILTX
  init_args:
    base_url: http://127.0.0.1:8188
    workflow_json_path: workflows/ltx2_3.json
    max_frames: 6
    input_node_descriptions:
      - node_id: "10"
        field: "prompt"
        source: prompt
      - node_id: "11"
        field: "image1"
        source: frame
        index: 0
      - node_id: "12"
        field: "image2"
        source: frame
        index: 1
      - node_id: "13"
        field: "image3"
        source: frame
        index: 2
      - node_id: "14"
        field: "image4"
        source: frame
        index: 3
      - node_id: "15"
        field: "image5"
        source: frame
        index: 4
      - node_id: "16"
        field: "image6"
        source: frame
        index: 5
```

## 3. 数据源类型说明

| source 类型 | 说明 | 示例 |
|------------|------|------|
| `prompt` | 文本提示词 | 用于 LLM 生成的描述文本 |
| `reference_image` | 参考图片（多张） | 角色肖像、风格参考 |
| `frame` | 帧图片（按索引） | 视频生成的首尾帧和中间帧 |
| `seed` | 随机种子 | 控制生成的可重复性 |
| `steps` | 采样步数 | 影响生成质量 |
| `cfg` | CFG 值 | 控制生成遵循提示的程度 |
| `width` | 输出宽度 | 图像/视频分辨率 |
| `height` | 输出高度 | 图像/视频分辨率 |

## 4. 中间帧生成策略

当镜头中存在多角色时，系统会自动规划中间帧：

```
镜头时长: 5 秒
首帧 (t=0s): 只有角色 A
中间帧 (t=2s): 角色 B 入场
尾帧 (t=5s): 角色 A 和 B 同时出现
```

系统会：
1. 分析镜头中角色的出现/退场时机
2. 计算需要生成中间帧的时间点
3. 为每个中间帧生成描述
4. 调用图片生成器生成中间帧图片
5. 将所有帧传给 LTX 进行视频生成

## 5. Prompt 转换

### 5.1 LTX 格式要求

LTX 2.3 不支持对话格式（如 `Alice: Hello`），需要转换为场景描述：

**错误格式：**
```
Alice: 你好 Bob！
Bob: 你好 Alice！
```

**正确格式：**
```
A cinematic scene in a cozy cafe with warm lighting.
Alice turns and smiles warmly, saying "你好 Bob！"
Bob waves back cheerfully, responding "你好 Alice！"
```

### 5.2 转换规则

| 原格式 | 转换后 |
|--------|--------|
| `Alice (Happy): 你好` | `Alice smiles happily, saying "你好"` |
| `Bob (Surprised): 什么？` | `Bob's eyes widen in surprise, exclaiming "什么？"` |
| `[Sound Effect] 门铃` | `with a doorbell sound in the background` |

## 6. 完整配置示例

```yaml
# configs/script2video_comfyui.yaml

chat_model:
  init_args:
    model: doubao-1-5-pro-32k-250115
    model_provider: openai
    api_key: <YOUR_API_KEY>
    base_url: https://ark.cn-beijing.volces.com/api/v3

image_generator:
  class_path: tools.ImageGeneratorComfyUIFlux
  init_args:
    base_url: http://127.0.0.1:8188
    workflow_json_path: workflows/flux_klein.json
    input_node_descriptions:
      - node_id: "3"
        field: "text_g"
        source: prompt
      - node_id: "4"
        field: "text_l"
        source: prompt
      - node_id: "5"
        field: "image"
        source: reference_image
        max_count: 4

video_generator:
  class_path: tools.VideoGeneratorComfyUILTX
  init_args:
    base_url: http://127.0.0.1:8188
    workflow_json_path: workflows/ltx2_3.json
    max_frames: 6
    input_node_descriptions:
      - node_id: "10"
        field: "prompt"
        source: prompt
      - node_id: "11"
        field: "image1"
        source: frame
        index: 0
      - node_id: "12"
        field: "image2"
        source: frame
        index: 1
      - node_id: "13"
        field: "image3"
        source: frame
        index: 2
      - node_id: "14"
        field: "image4"
        source: frame
        index: 3
      - node_id: "15"
        field: "image5"
        source: frame
        index: 4
      - node_id: "16"
        field: "image6"
        source: frame
        index: 5

working_dir: .working_dir/comfyui_video
```

## 7. 调试技巧

### 7.1 检查工作流是否正确加载

```python
from tools import ImageGeneratorComfyUIFlux

generator = ImageGeneratorComfyUIFlux(
    base_url="http://127.0.0.1:8188",
    workflow_json_path="workflows/flux_klein.json",
)
print(f"Loaded workflow keys: {list(generator.runner.workflow.keys())}")
```

### 7.2 测试单个生成

```python
import asyncio

async def test():
    from tools import ImageGeneratorComfyUIFlux
    
    generator = ImageGeneratorComfyUIFlux(
        base_url="http://127.0.0.1:8188",
        workflow_json_path="workflows/flux_klein.json",
    )
    
    image = await generator.generate_single_image(
        prompt="A cute cat sitting on a windowsill",
        reference_image_paths=[],
    )
    
    image.save("test_output.png")

asyncio.run(test())
```

### 7.3 常见问题

**Q: 提示 "No image/video output generated"**
- 检查工作流是否有正确的输出节点
- 确认 output_node_ids 是否正确配置

**Q: 帧图片没有正确传递**
- 检查 node_id 和 index 是否与工作流匹配
- 确认帧图片路径是否正确

**Q: Prompt 格式错误**
- 使用 PromptConverter 测试 prompt 转换
- 查看转换后的 prompt 是否符合 LTX 格式要求
