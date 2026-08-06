"""
角色音色代理 (VoiceDesigner)

根据 CharacterInScene 信息，通过 LLM 生成角色音色设计方案，
然后调用 ComfyUI Qwen TTS 设计音色并注册。

流程：
1. 接收 CharacterInScene（角色名、静态特征、动态特征）
2. LLM 生成 → VoiceDesign（音色提示词、代表性台词、角色描述、性别）
3. 调用 audio_generator.design_voice() 生成音色文件
4. 调用 register_voice() 注册音色到 VoiceManager
"""

import asyncio
import logging
import os
from typing import Any, Optional

from tenacity import retry, stop_after_attempt

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel
from utils.provider_presets import create_chat_model

from utils.retry import after_func
from utils.completion_logger import log_agent
from utils.voice import register_voice
from interfaces.character import CharacterInScene
from configs.config import working_dir

logger = logging.getLogger(__name__)


# ==================================================
# BaseModel：结构化输出
# ==================================================

class VoiceDesign(BaseModel):
    """角色音色设计方案"""

    voice_prompt: str = Field(
        description="音色生成提示词，详细描述角色的音色特点，用于AI TTS系统生成独特声音。"
                    "包括：音色质感(清澈/沙哑/温暖/冷峻等)、音高(高/中/低)、"
                    "语速(快/慢/适中)、情感基调(平静/激昂/忧郁/威严等)、说话风格(正式/随意/诗意等)"
    )
    character_text: str = Field(
        description="角色代表性台词，一段简洁的对白（建议15-50字），"
                    "用于展示角色的语言风格和声音特点，同时也是音色设计的参考文本"
    )
    character_description: str = Field(
        description="角色综合描述，概括包含以下内容："
                    "1.音色信息 2.身份背景 3.外貌特征 4.性格特质"
    )
    gender: str = Field(
        description="角色性别: Male 或 Female"
    )


# ==================================================
# Prompt
# ==================================================

SYSTEM_PROMPT = """
[ROLE]

You are a professional AI Voice Casting Director specialized in:

- voice personality design for AI TTS systems
- character voice archetype analysis
- timbre, pitch, tone, and speech tempo design
- character identity synthesis from visual and behavioral traits
- cinematic voice casting for animated/story characters

Your task is to design a distinctive voice for a character based on their profile.

==================================================
[INPUT]

The user will provide the following information:
- CHARACTER_NAME: 角色名称，如“小明”。within <CHARACTER_NAME> and </CHARACTER_NAME> tags
- STATIC_FEATURES: The static features of the character in this specific scene, such as facial features and body shape that remain constant or are rarely changed. within <STATIC_FEATURES> and </STATIC_FEATURES> tags
- DYNAMIC_FEATURES: he dynamic features of the character in this specific scene, such as clothing and accessories that may change from scene to scene. within <DYNAMIC_FEATURES> and </DYNAMIC_FEATURES> tags

==================================================
[CORE OBJECTIVE]

The designed voice must:

- match the character's physical appearance (age, build, facial features)
- reflect the character's personality traits
- embody the character's identity and background
- support appropriate emotional range
- be distinctive and recognizable
- be suitable for AI TTS generation

==================================================
[VOICE DESIGN DIMENSIONS]

Describe the character's voice across these dimensions:

1. **Timbre**: clear / raspy / warm / cold / soft / rough / magnetic / ethereal
2. **Pitch**: high / low / medium / slightly high / slightly low
3. **Tempo**: fast / slow / moderate / rhythmic
4. **Emotional tone**: calm / energetic / melancholic / authoritative / lively / gentle / serious / humorous
5. **Speech style**: formal / casual / poetic / concise / verbose / refined / bold
6. **Age feel**: child / teenager / young adult / middle-aged / elderly
7. **Special traits**: accent, habitual endings, laughter style, etc.

==================================================
[CHARACTER DESCRIPTION REQUIREMENTS]

The character_description must include these four sections:

1. **Voice Info**: summary of the character's vocal characteristics
2. **Identity & Background**: profession, social status, story context
3. **Appearance**: synthesized from static and dynamic features
4. **Personality**: traits inferred from the profile

==================================================
[VOICE PROMPT GUIDELINES]

The voice_prompt should:
- focus on vocal qualities, NOT visual descriptions
- be specific and actionable for TTS generation
- include tone, pace, emotion, and timbre details

Example: "A warm, gentle female voice with a medium pitch, speaking slowly and softly, conveying a sense of wisdom and calm authority. The voice has a maternal quality with slight melodic intonation."

==================================================
[OUTPUT]

{format_instructions}

**The language of output should be same as the input.**
"""

# [ROLE]

# 你是一名专业的AI语音选角导演，擅长于：

# - AI TTS系统的语音个性设计
# -人物声音原型分析
# -音色，音高，音调和语音节奏设计
# -从视觉和行为特征合成的角色身份
# -动画/故事角色的电影配音

# 你的任务是根据角色的形象设计一个独特的声音。

# ==================================================
# (核心目标)

# 设计的声音必须：

# 匹配角色的外表（年龄，体型，面部特征）
# -反映角色的个性特征
# -体现角色的身份和背景
# -支持适当的情绪范围
# -与众不同，可识别
# -适合AI TTS生成

# ==================================================
# [声音设计尺寸]

# 从这些方面描述角色的声音：

# 1. **音色**：清晰/刺耳/温暖/寒冷/柔软/粗糙/磁性/空灵
# 2. **音高**：高/低/中/稍高/稍低
# 3. 节奏：快/慢/中/有节奏
# 4. **情感基调：冷静/精力充沛/忧郁/权威/活泼/温柔/严肃/幽默
# 5. **演讲风格**：正式/随意/诗意/简洁/冗长/精炼/大胆
# 6. **年龄感受**：儿童/青少年/青年/中年/老年人
# 7. **特点：口音、习惯性结尾、笑声风格等。

# ==================================================
# [字符描述要求]

# character_description必须包含以下四个部分：

# 1. **声音信息**：角色声音特征的总结
# 2. **身份与背景**：职业、社会地位、故事背景
# 3. **外观**：由静态和动态特征合成
# 4. **个性**：从档案中推断出的特征

# ==================================================
# [语音提示指引]

# voice_prompt应该：

# -为更好的TTS兼容性使用英文
# -关注声音的质量，而不是视觉描述
# - TTS生成的具体和可操作
# -包括语气，节奏，情感和音色细节

# 例如：“一个温暖、温柔、中等音高的女性声音，说话缓慢而温柔，传达出一种智慧和冷静的权威感。”她的声音带有轻微的旋律语调，带有母性。”

# ==================================================
# (输出)

# {format_instructions}

HUMAN_PROMPT = """
<CHARACTER_NAME>
{character_name}
</CHARACTER_NAME>

<STATIC_FEATURES>
{static_features}
</STATIC_FEATURES>

<DYNAMIC_FEATURES>
{dynamic_features}
</DYNAMIC_FEATURES>
"""


# ==================================================
# Agent
# ==================================================

class VoiceDesigner:
    """AI 角色音色设计师：为角色设计独特音色并注册"""

    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = create_chat_model(
            model_provider="qwen",
            model="deepseek-v4-flash-0731",
        )

    # ---------- LLM：生成音色设计方案 ----------

    @log_agent("VoiceDesigner")
    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_voice_llm(
        self,
        character: CharacterInScene,
        retry_timeout: int = 500,
    ) -> VoiceDesign:
        """
        使用 LLM 生成角色音色设计方案

        Args:
            character: 场景中的角色（姓名、静态特征、动态特征）
            retry_timeout: LLM 调用超时（秒）

        Returns:
            VoiceDesign: 含 voice_prompt / character_text / character_description / gender
        """
        parser = PydanticOutputParser(pydantic_object=VoiceDesign)

        messages = [
            ('system', SYSTEM_PROMPT.format(
                format_instructions=parser.get_format_instructions(),
            )),
            ('human', HUMAN_PROMPT.format(
                character_name=character.identifier_in_scene,
                static_features=character.static_features,
                dynamic_features=character.dynamic_features or "(none)",
            )),
        ]

        chain = self.chat_model | parser
        response: VoiceDesign = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )

        logger.info(
            "Voice design LLM: '%s' → gender=%s, text=%s",
            character.identifier_in_scene,
            response.gender,
            response.character_text[:30],
        )
        return response

    # ---------- 完整流程：LLM + 工具调用 + 注册 ----------

    async def design_and_register(
        self,
        character: CharacterInScene,
        audio_generator: Any,
    ) -> dict:
        """
        完整音色设计与注册流程：

        1. LLM 生成 VoiceDesign 方案
        2. 调用 audio_generator.design_voice() 生成音色文件
        3. 调用 audio_generator.register_voice() 保存音色到 ComfyUI
        4. 调用 register_voice() 注册到 VoiceManager

        Args:
            character: 场景中的角色信息
            audio_generator: AudioGeneratorComfyUIQwenTTS 实例

        Returns:
            str: 注册的音色文件名
        """
        character_name = character.identifier_in_scene
        audio_path = os.path.join(working_dir, "voices", f"{character_name}.mp3")

        if os.path.exists(audio_path):
            logger.info("Voice audio exists: %s", character_name)
            return {
                "character": character_name,
                "gender": "unknown",
            }

        # ── Step 1: LLM 生成音色设计方案 ──
        logger.info("Step 1/3: LLM designing voice for '%s'...", character_name)
        voice_design = await self.design_voice_llm(character)

        # ── Step 2: ComfyUI 生成音色文件 ──
        logger.info("Step 2/3: Generating voice audio for '%s'...", character_name)
        voice_output = await audio_generator.design_voice(
            character=character_name,
            prompt=f"{voice_design.voice_prompt}\n{voice_design.character_description}",
            text=voice_design.character_text,
        )

        
        voice_output.save(audio_path)
        logger.info("Voice audio generated: %s", audio_path)

        # ── Step 3: 保存音色到 ComfyUI + 注册到 VoiceManager ──
        logger.info("Step 3/3: Saving and registering voice for '%s'...", character_name)
        await audio_generator.register_voice(
            character=character_name,
            text=voice_design.character_text,
            audio_path=audio_path,
        )

        logger.info(
            "Voice registered: '%s' → voice_file='%s', gender=%s",
            character_name,
            character_name,
            voice_design.gender,
        )

        return {
            "character": character_name,
            "gender": voice_design.gender,
        }
