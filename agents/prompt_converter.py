"""
PromptConverter: Prompt 格式转换器

将不同格式的 prompt 转换为适合目标模型的格式

主要转换场景：
1. 对话格式 → LTX 2.3 场景描述格式
   输入: "Alice (Happy): Hello, how are you?"
         "Bob (Curious): I'm fine, thank you!"
   输出: "A cinematic scene. Alice smiles happily, saying "你好". Bob leans forward, curious, responding "我很好，谢谢"."

2. 保持英文场景描述，中文对话内容
   输入: "[Sound Effect] Ambient cafe music. Alice: 你好 Bob"
   输出: "A cinematic scene in a cozy cafe with soft ambient music. Alice turns and smiles warmly, saying "你好 Bob"."
"""

import logging
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import resolve_chat_model_config


logger = logging.getLogger(__name__)

system_prompt_template_convert_dialogue = \
"""
[Role]
你是一个专业的提示转换器，专门用于将基于对话的视频脚本转换为针对LTX video 2.3模型优化的电影场景描述。

[Task]
你的任务是将音频描述（包含音效、对话等信息）融合到场景视觉描述中。

1. **理解场景描述和音频描述**: 阅读并理解场景描述和音频描述，思考对应的音效、对话在视频场景的哪个时间段引入最合理。
2. **Dialogue to Scene Description**: Convert dialogue format (e.g., "Alice: Hello" or "Alice (Happy): Hello") into narrative scene descriptions.
  - Instead of: "Alice: 你好"
  - Use: "Alice turns and smiles warmly, saying '你好'"

3. **Remove Colons in Dialogue**: LTX 2.3 does not support the colon format. Replace "Name: dialogue" with descriptive action sentences.
  - Wrong: "Alice: Hello Bob"
  - Correct: "Alice waves to Bob, 'Hello Bob'"

[Input]
你将收到：
- Audio description: A detailed description of the audio in the shot. The audio input is enclosed within <Audio> and </Audio>.
- Motion description: The motion description of the shot. Describe the dynamic visual changes within the shot (camera movement and the movement of elements within the frame). The motion input is enclosed within <Motion> and </Motion>
- duration: The duration of the shot in seconds(include). The duration input is enclosed within <Duration> and </Duration>

[Output Format]
{format_instructions}

[Guidelines]
- 要保持场景视觉描述的意思不发生改变
- 描述使用中文
- **确保最终场景描述融入音频（包含音效、对话、对话情绪）**
- **音频融入要自然，位置要正确**: 确保融入正确的位置，不要一味的加在最后
- 要保留人物外貌和动态特征描述（如穿着）
- 对话融入要保留标点符号，用“”括起来
- 确保句子之间的逻辑流畅
- **计算对话时长**：按对话语速 4.5 字/秒 （含自然停顿）来计算
- 如果有多个对话，对话之间需有停顿，建议间隔0.5秒

请严格遵循以上规则，直接输出 JSON，不要添加解释。
"""

human_prompt_template_convert_dialogue = \
"""
<Audio>
{audio_desc}
</Audio>

<Motion>
{motion_desc}
</Motion>

<Duration>
{duration}
</Duration>
"""

class Dialogue(BaseModel):
    speaker: str = Field(..., description="对话者名称")
    gender: Optional[str] = Field(default="Female", description="对话者性别。 枚举 Male、Female")
    emotion: str = Field(..., description="对话者情绪")
    dialogue: str = Field(..., description="对话内容")
    start_time: float = Field(..., description="对话开始时间（秒）")
    duration: float = Field(..., description="对话时长（秒）")

class ShotDescriptionWithDialogues(BaseModel):
    prompt: str = Field(..., description="融入音效的场景视觉描述")
    dialogues: Optional[List[Dialogue]] = Field(default=[], description="对话列表")
    shot_duration: Optional[float] = Field(..., description="镜头时长（秒）")
    use_xianxia_lora: Optional[bool] = Field(default=False, description="是否使用仙侠lora。如果有对话，请设置为True")

class PromptConverter:
    """
    Prompt 转换器
    
    负责将对话格式的 prompt 转换为适合 LTX 2.3 的场景描述格式
    """
    
    def __init__(
        self,
    ):
        """
        初始化 Prompt 转换器
        
        Args:
            chat_model: 可选的 LLM 模型用于复杂转换
            use_llm_conversion: 是否使用 LLM 进行转换（复杂场景），否则使用规则转换
        """
        config = resolve_chat_model_config(
            {
                "model_provider": "qwen",
                "model": "deepseek-v4-pro",
            }
        )
        self.chat_model = init_chat_model(**config)

    async def convert(
        self,
        audio_desc: str,
        motion_desc: str = "",
        shot_duration: float = 5.0,
        retry_timeout: int = 300,
    ) -> ShotDescriptionWithDialogues:
        """
        转换 prompt 为 LTX 2.3 格式
        将音频描述融入到动作描述中，使对话出现在正确的场景位置。
        
        Args:
            audio_desc: 音频描述（对话）
            motion_desc: 动作描述（镜头运动和角色动作）
            shot_duration: 镜头时长（秒）
        
        Returns:
            LTX 格式的 prompt，对话已融入动作描述中
        """
        
        parser = PydanticOutputParser(pydantic_object=ShotDescriptionWithDialogues)
        messages = [
            SystemMessage(content=system_prompt_template_convert_dialogue.format(
                format_instructions=parser.get_format_instructions(),
            )),
            HumanMessage(content=human_prompt_template_convert_dialogue.format(
                audio_desc=audio_desc or "(No audio)",
                motion_desc=motion_desc or "(No motion description)",
                duration=shot_duration,
            )),
        ]
        
        chain = self.chat_model | parser
        result = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        result.shot_duration = shot_duration
        
        return result
            
