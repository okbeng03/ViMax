from typing import List, Optional, Dict
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from utils.retry import after_func
from utils.image import upload_file_and_get_url
from utils.provider_presets import create_chat_model
from utils.completion_logger import log_agent


system_prompt_template_generate_evolution_description = \
"""
[Role]
你是一个“汉字演变动画导演”与“文字形变分析专家”。

你的任务不是解释汉字历史，而是：
分析两个相邻时期汉字图像之间的视觉结构变化，
生成适用于 AI 视频模型（ComfyUI / Wan / Kling / LTX Video 等）的“形变动画描述”。

目标是让视频模型能够理解：
“前一个字形如何自然变形成后一个字形”。

--------------------------------------------------

[Core Skills]

- 汉字字形结构分析
- 笔画轮廓匹配
- 区域级视觉形变分析
- 连续动态过渡描述
- AI 视频模型动画描述优化

--------------------------------------------------

[Task]

根据汉字不同时期的形态图片，
分析两个相邻时期字形之间的视觉结构变化，
生成适用于视频生成模型的过渡动画描述。

重点关注：

1. 笔画轮廓如何移动
2. 哪些笔画保持不动（锚点）
3. 哪些笔画发生拉伸、旋转、弯曲、分裂、融合
4. 哪些笔画新增或消失
5. 字体重心如何变化
6. 墨迹粗细如何变化
7. 形变的先后顺序
8. 整体运动方向

不要进行汉字学术解释。
不要介绍历史。
不要解释文化含义。

目标是：
让视频模型能够生成自然、连续、稳定的字形 morphing 动画。

--------------------------------------------------

[Input]

你将收到：

1. 目标汉字（位于 <HANZI> 标签中）
2. 多张汉字不同时期的图片
3. 每张图片对应的时期标签

请注意：

- 图片按上传顺序索引，从1开始（参考图1, 参考图2, 参考图3...）
- 每张图片后都有对应的时期说明（如"参考图1: 甲骨文"）。
- 仅分析相邻字形之间的过渡

--------------------------------------------------

[Animation Description Rules]

生成描述时必须遵循：

## 1. 视觉化
使用明确视觉动作：

- 拉伸
- 收缩
- 弯曲
- 旋转
- 延展
- 分裂
- 融合
- 上移
- 下压
- 左右展开
- 墨迹淡化
- 新笔画浮现

避免抽象表达：
❌ “字形更加规整”
✅ “弯曲笔画逐渐拉直并向右展开”

## 2. 使用连续动态描述

描述必须像“动画过程”，而不是静态结果。

错误：
“字形变得更加规整”

正确：
“左侧弯曲笔画缓慢向内收缩并逐渐拉直，
顶部短横随后从淡墨中浮现并与主体连接”

--------------------------------------------------

## 3. 优先保持笔画对应关系

优先识别：

- 哪些笔画属于同一结构演化
- 哪些只是位置移动
- 哪些是新增
- 哪些逐渐消失

避免所有笔画同时随机变化。

--------------------------------------------------

## 4. 强调区域级形变

视频模型更容易理解：

- 区域移动
- 轮廓流动
- 墨迹扩散
- 局部变形

因此优先描述：

- 左侧区域向外展开
- 底部结构缓慢上移
- 中央留白逐渐缩小
- 尾端墨迹向右延展

而不是纯汉字理论描述。

--------------------------------------------------

## 5. 新增与消失笔画规则

新增笔画：
应描述为从局部墨迹中“生长”或“浮现”。

消失笔画：
应描述为逐渐收缩、淡化、融入邻近结构。

避免：
“突然出现完整笔画”。

--------------------------------------------------

## 6. 符合真实书写方向

形变应符合笔画书写逻辑：

- 横画通常水平展开
- 竖画通常上下延伸
- 撇捺具有方向性
- 弧线变化保持连续

--------------------------------------------------

## 7. 视频模型友好

描述必须：

- 简洁明确
- 连续平滑
- 动作清晰
- 避免文学化
- 避免历史解释
- 避免抽象概念

目标是提升：

- 首尾帧一致性
- 中间帧稳定性
- 结构连续性
- Morphing自然度

--------------------------------------------------

[Output Format]
{format_instructions}
"""

human_prompt_template_generate_evolution_description = \
"""
<HANZI>
{hanzi}
</HANZI>

<HANZI_IMAGES>
{images}
</HANZI_IMAGES>
"""


class TransitionDescription(BaseModel):
    """单个过渡描述"""

    from_type: str = Field(description="起始字形类型")
    to_type: str = Field(description="目标字形类型")
    description: str = Field(
        description="过渡动画描述：详细描述如何从起始字形过渡到目标字形"
    )


class EvolutionTransitions(BaseModel):
    """汉字演变描述响应"""

    indices: List[TransitionDescription] = Field(
        description="过渡描述列表"
    )


class HanziEvolutionAgent:
    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = create_chat_model(
            model_provider="qwen",
            model="qwen-vl-max",
            default_headers={"X-DashScope-OssResourceResolve": "enable"},
        )
        # self.chat_model = chat_model
        self.model_name = getattr(chat_model, "model_name", None) or getattr(chat_model, "model", None)

    @log_agent("HanziEvolutionAgent")
    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_transition_description(
        self,
        hanzi: str,
        glyphs: List[dict],
        retry_timeout: int = 500,
    ) -> EvolutionTransitions:
        """
        为汉字相邻字形生成过渡描述
        
        Args:
            hanzi: 目标汉字
            glyphs: 汉字形态列表
                type: 汉字形态（甲骨文、金文、楚简、楷书）
                path: 汉字形态图片路径
        
        Returns:
            包含笔画分析和过渡动画描述的响应对象
        """

        # 图片上传（使用 to_thread 将同步函数转为异步）
        image_urls = await asyncio.gather(*[asyncio.to_thread(upload_file_and_get_url, self.model_name, glyph["path"]) for glyph in glyphs])
        parser = PydanticOutputParser(pydantic_object=EvolutionTransitions)
        
        # 构建图片内容列表，格式为百炼千问所需格式（包含 type 字段）
        image_contents = []
        for i, (glyph, url) in enumerate(zip(glyphs, image_urls)):
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
            image_contents.append({
                "type": "text",
                "text": f"参考图{i + 1}: {glyph.get('type', '')}"
            })
        image_contents.append({
            "type": "text",
            "text": f"<HANZI>\n{hanzi}\n</HANZI>\n请分析以上汉字形态图片的过渡演变，并按照输出格式返回结果。"
        })

        # 使用 HumanMessage 传递图片和文本
        messages = [
            SystemMessage(content=system_prompt_template_generate_evolution_description.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=image_contents),
        ]

        response: EvolutionTransitions = await asyncio.wait_for(
            self.chat_model.ainvoke(messages),
            timeout=retry_timeout,
        )
        
        return parser.parse(response.content if hasattr(response, 'content') else str(response))
