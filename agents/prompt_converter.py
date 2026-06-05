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
from interfaces import CharacterInScene
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import resolve_chat_model_config


logger = logging.getLogger(__name__)

system_prompt_template_convert_dialogue = \
"""
[Role]

你是一个专业的提示转换器，专门用于将基于对话的视频脚本转换为针对 LTX Video 2.3 模型优化的电影场景描述。

---

[Task]

你的任务是将音频描述（包含音效、对话、语气等信息）和角色设定信息融合到镜头视觉描述中，生成适合视频生成模型理解的完整场景描述。

你需要：

1. **理解场景描述和音频描述**: 阅读并理解场景描述和音频描述，思考对应的音效、对话在视频场景的哪个时间段引入最合理。
2. **Dialogue to Scene Description**: Convert dialogue format (e.g., "Alice: Hello" or "Alice (Happy): Hello") into narrative scene descriptions.
  - Instead of: "Alice: 你好"
  - Use: "Alice turns and smiles warmly, saying '你好'"

3. **Remove Colons in Dialogue**: LTX 2.3 does not support the colon format. Replace "Name: dialogue" with descriptive action sentences.
  - Wrong: "Alice: Hello Bob"
  - Correct: "Alice waves to Bob, 'Hello Bob'"
4. 理解角色设定信息。
5. 将角色设定补充到镜头中的角色描述里。如果镜头里多次出现角色，只将设定补充在第一个位置。
6. 保持场景逻辑、时间顺序和动作连续性。
7. 输出最终优化后的场景描述。

---

[Input]
你将收到：
- Audio description: A detailed description of the audio in the shot. The audio input is enclosed within <Audio> and </Audio>.
- Motion description: The motion description of the shot. Describe the dynamic visual changes within the shot (camera movement and the movement of elements within the frame). The motion input is enclosed within <Motion> and </Motion>
- duration: The duration of the shot in seconds(include). The duration input is enclosed within <Duration> and </Duration>
- characters: A list describing basic information for each character, such as name, personality traits, appearance (if relevant). The characters input is enclosed within <Characters> and </Characters>

---

[Output Format] 
{format_instructions}

---

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

---

[Character Profile Injection Rules]

当镜头中出现角色时，必须根据 Characters 补充角色外观描述。

格式：

角色(
静态外观特征
动态外观特征
)

Rules:

- 保留 Motion Description 原有动作
- 从 Character 提取静态特征
- 不要改变原场景含义
- 不要增加剧本中不存在的新动作
- 不要增加新情绪
- 不要增加新道具
- 不要增加新场景元素

Character 仅用于补充角色外观一致性

示例：

原文：

一位身着红色盘扣小马甲与黄色灯笼裤的小男孩仰头凝望。

改写：

一位小男孩(
5岁，
两个冲天揪发型，
圆润脸颊，
大而明亮的眼睛，
红色盘扣小马甲，
黄色灯笼裤，
黑色布鞋
)仰头凝望。


请严格遵循以上规则，直接输出 JSON，不要添加任何解释。
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

<Characters>
{characters_str}
</Characters>
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
    is_small_people: Optional[bool] = Field(default=False)

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
                "model": "qwen3.6-27b",
            }
        )
        self.chat_model = init_chat_model(**config)

    async def convert(
        self,
        audio_desc: str,
        motion_desc: str = "",
        characters: List[CharacterInScene] = [],
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
        characters_str = "\n".join([f"{character.identifier_in_scene}: {character.static_features}{character.dynamic_features}" for character in characters])

        messages = [
            SystemMessage(content=system_prompt_template_convert_dialogue.format(
                format_instructions=parser.get_format_instructions(),
            )),
            HumanMessage(content=human_prompt_template_convert_dialogue.format(
                audio_desc=audio_desc or "(No audio)",
                motion_desc=motion_desc or "(No motion description)",
                duration=shot_duration,
                characters_str=characters_str
            )),
        ]
        
        chain = self.chat_model | parser
        result = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        result.shot_duration = shot_duration
        
        return result
            
