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

TODO:: 基于规则的基本不可用，太死了；另外prompt是不是缺少了要将对话融入场景描述，现在按json结构输出再合并。
"""

import re
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import resolve_chat_model_config


logger = logging.getLogger(__name__)


# ============= Prompt 转换模式 =============

# 对话模式匹配
# 例如: "Alice (Happy): Hello" 或 "Alice: Hello" 或 "[Speaker] Alice (Happy): Hello"
DIALOGUE_PATTERNS = [
    # [Speaker] Name (Emotion): Dialogue
    r'\[Speaker\]\s*([^\(]+?)\s*(?:\(([^)]+)\))?\s*:\s*(.+?)(?=(?:\n\[Speaker\])|$)',
    # Name (Emotion): Dialogue
    r'^([^\(]+?)\s*(?:\(([^)]+)\))?\s*:\s*(.+?)(?=(?:\n[A-Z][^\(]+)|$)',
]

# 音效模式匹配
SOUND_EFFECT_PATTERN = r'\[Sound Effect\]\s*(.+?)(?=\n|$)'

# 旁白模式匹配
NARRATION_PATTERN = r'\[Narration\]\s*[:：]?\s*(.+?)(?=\n|$)'


class DialogueEntry(BaseModel):
    """单条对话条目"""
    speaker: str = Field(description="说话者名称")
    emotion: Optional[str] = Field(default=None, description="说话时的情绪")
    dialogue: str = Field(description="对话内容")
    line_number: Optional[int] = Field(default=None, description="原始文本中的行号")


class ParsedAudioInfo(BaseModel):
    """解析后的音频信息"""
    dialogues: List[DialogueEntry] = Field(description="对话列表")
    sound_effects: List[str] = Field(description="音效描述列表")
    narrations: List[str] = Field(description="旁白列表")


class LTXPromptFormat(BaseModel):
    """LTX 2.3 格式的 prompt"""
    scene_setting: str = Field(description="场景设定（英文）")
    actions: List[str] = Field(description="动作/对话描述列表（英文场景 + 中文对话内容）")
    atmosphere: str = Field(description="氛围描述")


system_prompt_template_convert_dialogue = \
"""
[Role]
You are a professional prompt converter specialized in transforming dialogue-based video scripts into cinematic scene descriptions optimized for LTX Video 2.3 model.

[Task]
Your task is to convert dialogue-format prompts into LTX 2.3 compatible scene descriptions with the following key transformations:

1. **Dialogue to Scene Description**: Convert dialogue format (e.g., "Alice: Hello" or "Alice (Happy): Hello") into narrative scene descriptions.
   - Instead of: "Alice: 你好"
   - Use: "Alice turns and smiles warmly, saying '你好'"

2. **Bilingual Format**: Keep scene descriptions in English, but embed Chinese dialogue content within the English narrative.
   - Scene: "A cozy cafe with warm lighting"
   - Dialogue: "Alice waves and greets, '咖啡馆开张了！'"

3. **Remove Colons in Dialogue**: LTX 2.3 does not support the colon format. Replace "Name: dialogue" with descriptive action sentences.
   - Wrong: "Alice: Hello Bob"
   - Correct: "Alice waves to Bob, 'Hello Bob'"

4. **Emotion Mapping**: Map emotion tags to appropriate action descriptions.
   - (Happy) → smiles, laughs, waves happily
   - (Sad) → sighs, looks down, wipes tears
   - (Angry) → frowns, clenches fists, raises voice
   - (Surprised) → gasps, eyes widen, jumps back
   - (Curious) → leans forward, tilts head, raises eyebrow
   - (Thinking) → pauses, looks thoughtful, rubs chin

5. **Cinematic Elements**: Add appropriate scene context and atmosphere.

[Input]
You will receive:
- Original audio descriptions from video scripts (dialogues, sound effects, narrations)
- Scene visual descriptions

[Output]
{format_instructions}

[Guidelines]
- Output language: Scene descriptions in English, dialogue content should remain in its original language (Chinese)
- Keep the output concise but descriptive
- Focus on visualizable actions and expressions
- Ensure logical flow between sentences
- Maximum 5-6 key actions for a 5-second shot
"""

# [Role]
# 你是一个专业的提示转换器，专门用于将基于对话的视频脚本转换为针对LTX video 2.3模型优化的电影场景描述。

# (任务)
# 你的任务是通过以下关键转换将对话框格式的提示转换成ltx2.3兼容的场景描述：

# 1. **对话到场景描述**：将对话格式（例如，“Alice: Hello”或“Alice (Happy): Hello”）转换为叙事场景描述。
# ——而不是：“Alice：“
# -使用：“Alice转过身来，热情地笑了，说了声‘亲亲的’。”

# 2. **双语格式：场景描述保持英文，但在英文叙事中嵌入中文对话内容。
# -场景：“舒适的咖啡厅，温暖的灯光”
# -对话：“爱丽丝挥手打招呼，”’”

# 3. **删除对话框中的冒号**:ltx2.3不支持冒号格式。用描述性的动作句代替“姓名：对话”。
# 错误示例：“Alice: Hello Bob”
# 正确示例：“爱丽丝向鲍勃挥手，‘你好，鲍勃’”

# 4. **情感映射**：将情感标签映射到适当的动作描述。
# -（快乐）→微笑，大笑，愉快地挥手
# -（悲伤）→叹气，低头，擦眼泪
# -（生气）→皱眉、握紧拳头、提高嗓门
# -（惊讶）→喘气，睁大眼睛，向后跳
# -（好奇）→身体前倾，倾斜头部，扬起眉毛
# -（思考）→停顿，若有所思，揉揉下巴

# 5. **电影元素**：添加适当的场景背景和氛围。

# (输入)
# 您将收到：
# -来自视频脚本的原始音频描述（对话，音效，叙述）
# -场景视觉描述

# (输出)
# {format_instructions}

# (指南)
# -输出语言：场景描述为英文，对话内容保持原语言（中文）
# —输出内容要简洁，但要有描述性
# -专注于可视化的动作和表达
# -确保句子之间的逻辑流畅
# - 5秒射击最多5-6个关键动作

human_prompt_template_convert_dialogue = \
"""
<ORIGINAL_AUDIO_DESC>
{audio_desc}
</ORIGINAL_AUDIO_DESC>

<ORIGINAL_VISUAL_DESC>
{visual_desc}
</ORIGINAL_VISUAL_DESC>

<ORIGINAL_MOTION_DESC>
{motion_desc}
</ORIGINAL_MOTION_DESC>

<SHOT_DURATION>
{shot_duration} seconds
</SHOT_DURATION>

[IMPORTANT]
Your task is to create a unified scene description that:
1. Starts with scene setting (from visual_desc)
2. Describes actions WITH dialogues integrated (from motion_desc + audio_desc combined)

Example transformation:
- motion_desc: "Camera pans slowly. Alice turns around."
- audio_desc: "Alice: Hello Bob!"
- WRONG: "Camera pans slowly. Alice turns around.\n[Speaker] Alice: Hello Bob!"
- CORRECT: "Camera pans slowly. Alice turns around and waves, saying 'Hello Bob!'"
"""


class PromptConverter:
    """
    Prompt 转换器
    
    负责将对话格式的 prompt 转换为适合 LTX 2.3 的场景描述格式
    """
    
    # 情绪词到动作的映射
    EMOTION_ACTION_MAP = {
        "happy": ["smiles warmly", "laughs happily", "waves cheerfully", "grins broadly"],
        "sad": ["looks downcast", "sighs softly", "wipes away tears", "frowns with sorrow"],
        "angry": ["furrows brow", "clenches fists", "raises voice", "scowls with frustration"],
        "surprised": ["gasps in shock", "eyes widen", "jumps back", "looks stunned"],
        "curious": ["leans forward", "tilts head", "raises eyebrow", "looks intrigued"],
        "thinking": ["pauses thoughtfully", "rubs chin", "looks contemplative", "stares into distance"],
        "nervous": ["fidgets nervously", "shifts weight", "bites lip", "twists hands"],
        "confused": ["scratches head", "looks puzzled", "furrows brow", "mouths '?'"],
        "tired": ["yawns softly", "rubs eyes", "slouches tiredly", "stretches wearily"],
        "excited": ["bounces excitedly", "claps hands", "jumps up", "shouts with joy"],
    }
    
    def __init__(
        self,
        use_llm_conversion: bool = True,
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
                "model": "qwen3.5-plus",
            }
        )
        self.chat_model = init_chat_model(**config)
        self.use_llm_conversion = use_llm_conversion
    
    def parse_audio_description(self, audio_desc: str) -> ParsedAudioInfo:
        """
        解析音频描述文本
        
        Args:
            audio_desc: 原始音频描述文本
        
        Returns:
            ParsedAudioInfo: 解析后的音频信息
        """
        dialogues = []
        sound_effects = []
        narrations = []
        
        lines = audio_desc.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 尝试匹配音效
            sound_match = re.search(SOUND_EFFECT_PATTERN, line, re.IGNORECASE)
            if sound_match:
                sound_effects.append(sound_match.group(1).strip())
                continue
            
            # 尝试匹配旁白
            narration_match = re.search(NARRATION_PATTERN, line, re.IGNORECASE)
            if narration_match:
                narrations.append(narration_match.group(1).strip())
                continue
            
            # 尝试匹配对话
            for pattern in DIALOGUE_PATTERNS:
                match = re.search(pattern, line, re.MULTILINE | re.DOTALL)
                if match:
                    speaker = match.group(1).strip()
                    emotion = match.group(2).strip() if match.group(2) else None
                    dialogue = match.group(3).strip()
                    
                    dialogues.append(DialogueEntry(
                        speaker=speaker,
                        emotion=emotion,
                        dialogue=dialogue,
                        line_number=i,
                    ))
                    break
        
        return ParsedAudioInfo(
            dialogues=dialogues,
            sound_effects=sound_effects,
            narrations=narrations,
        )
    
    def emotion_to_action(self, emotion: Optional[str]) -> str:
        """将情绪标签转换为动作描述"""
        if not emotion:
            return "says"
        
        emotion_lower = emotion.lower()
        
        # 精确匹配
        if emotion_lower in self.EMOTION_ACTION_MAP:
            actions = self.EMOTION_ACTION_MAP[emotion_lower]
            import random
            return random.choice(actions)
        
        # 部分匹配
        for key, actions in self.EMOTION_ACTION_MAP.items():
            if key in emotion_lower:
                import random
                return random.choice(actions)
        
        return "says"
    
    def rule_based_convert(
        self,
        audio_desc: str,
        visual_desc: str,
        shot_duration: float = 5.0,
    ) -> str:
        """
        基于规则的 prompt 转换
        
        适用于简单场景的快速转换
        
        Args:
            audio_desc: 音频描述
            visual_desc: 视觉描述
            shot_duration: 镜头时长（秒）
        
        Returns:
            LTX 格式的 prompt
        """
        # 解析音频信息
        parsed = self.parse_audio_description(audio_desc)
        
        # 构建场景描述
        parts = []
        
        # 1. 基础场景设置（从 visual_desc 提取）
        scene_setting = self._extract_scene_setting(visual_desc)
        if scene_setting:
            parts.append(scene_setting)
        
        # 2. 添加音效作为氛围
        if parsed.sound_effects:
            parts.append(f"with {', '.join(parsed.sound_effects)}")
        
        # 3. 转换对话为动作描述
        for dialogue in parsed.dialogues:
            action = self.emotion_to_action(dialogue.emotion)
            # 对话内容保持原语言
            action_desc = f"{dialogue.speaker} {action}, '{dialogue.dialogue}'"
            parts.append(action_desc)
        
        # 4. 添加旁白
        for narration in parsed.narrations:
            parts.append(f"Voiceover: {narration}")
        
        # 组合结果
        result = ". ".join(parts)
        
        # 添加结尾氛围
        if len(parts) <= 3:
            result += "."
        
        return result
    
    def _extract_scene_setting(self, visual_desc: str) -> Optional[str]:
        """从视觉描述中提取场景设定"""
        # 简化实现：取视觉描述的前半部分作为场景设定
        if not visual_desc:
            return None
        
        # 移除镜头类型描述（如 "Medium shot", "Close-up" 等）
        patterns_to_remove = [
            r'^(Extreme )?(Long|Wide|Medium|Close-up|Extreme close-up|Establishing) shot[.,]?',
            r"^(Low|High|Eye-level|Bird'?s-eye|Worm'?s-eye) angle[.,]?",
            r'^(Static|Dolly in|Dolly out|Pan|Tilt|Tracking|Handheld) [a-z]+[.,]?',
        ]
        
        desc = visual_desc
        for pattern in patterns_to_remove:
            desc = re.sub(pattern, '', desc, flags=re.IGNORECASE)
        
        # 取前 100 字符作为场景设定
        scene = desc.strip()[:150].rsplit(',', 1)[0]
        if scene:
            scene = scene[0].upper() + scene[1:]
        
        return scene if scene else None
    
    async def llm_convert(
        self,
        audio_desc: str,
        visual_desc: str,
        motion_desc: str = "",
        shot_duration: float = 5.0,
        retry_timeout: int = 300,
    ) -> str:
        """
        使用 LLM 进行 prompt 转换
        
        适用于复杂场景的精确转换。
        将对话融入场景叙述中，使对话出现在正确的动作位置。
        
        Args:
            audio_desc: 音频描述（对话）
            visual_desc: 视觉描述（场景设定）
            motion_desc: 动作描述（镜头运动和角色动作）
            shot_duration: 镜头时长（秒）
            retry_timeout: 超时时间
        
        Returns:
            LTX 格式的 prompt，对话已融入动作描述中
        """
        if not self.chat_model:
            logger.warning("No chat model provided, falling back to rule-based conversion")
            return self.rule_based_convert(audio_desc, visual_desc, shot_duration)
        
        parser = PydanticOutputParser(pydantic_object=LTXPromptFormat)
        
        messages = [
            SystemMessage(content=system_prompt_template_convert_dialogue.format(
                format_instructions=parser.get_format_instructions()
            )),
            HumanMessage(content=human_prompt_template_convert_dialogue.format(
                audio_desc=audio_desc or "(No audio)",
                visual_desc=visual_desc or "(No visual description)",
                shot_duration=shot_duration,
                motion_desc=motion_desc or "(No motion description)",
            )),
        ]
        
        import asyncio
        chain = self.chat_model | parser
        
        try:
            result: LTXPromptFormat = await asyncio.wait_for(
                chain.ainvoke(messages),
                timeout=retry_timeout,
            )
            
            # 组合为最终 prompt
            prompt_parts = [result.scene_setting]
            prompt_parts.extend(result.actions)
            prompt_parts.append(result.atmosphere)
            
            desc = ". ".join(filter(None, prompt_parts))
            logger.info(f"LLM conversion result: {desc}")
            return desc
            
        except Exception as e:
            logger.error(f"LLM conversion failed: {e}, falling back to rule-based")
            return self.rule_based_convert(audio_desc, visual_desc, shot_duration)
    
    async def convert(
        self,
        audio_desc: str,
        visual_desc: str,
        motion_desc: str = "",
        shot_duration: float = 5.0,
    ) -> str:
        """
        转换 prompt 为 LTX 2.3 格式
        
        自动选择转换方式：
        - 简单场景：规则转换
        - 复杂场景：LLM 转换
        
        将对话融入到动作描述中，使对话出现在正确的场景位置。
        
        Args:
            audio_desc: 音频描述（对话）
            visual_desc: 视觉描述（场景设定）
            motion_desc: 动作描述（镜头运动和角色动作）
            shot_duration: 镜头时长（秒）
        
        Returns:
            LTX 格式的 prompt，对话已融入动作描述中
        """
        # 检查是否需要 LLM 转换
        parsed = self.parse_audio_description(audio_desc)
        
        # 如果有多角色对话、复杂情绪、或旁白，使用 LLM
        needs_llm = (
            len(parsed.dialogues) > 2 or
            any(d.emotion for d in parsed.dialogues) or
            len(parsed.narrations) > 0 or
            self.use_llm_conversion and self.chat_model
        )
        
        if needs_llm and self.chat_model:
            return await self.llm_convert(audio_desc, visual_desc, motion_desc, shot_duration)
        else:
            return self.rule_based_convert(audio_desc, visual_desc, shot_duration)


# 便捷函数
async def convert_to_ltx_format(
    audio_desc: str,
    visual_desc: str,
    shot_duration: float = 5.0,
    chat_model=None,
) -> str:
    """
    将对话格式转换为 LTX 2.3 格式的便捷函数
    
    Args:
        audio_desc: 音频描述
        visual_desc: 视觉描述
        shot_duration: 镜头时长
        chat_model: 可选的 LLM 模型
    
    Returns:
        LTX 格式的 prompt
    """
    converter = PromptConverter(chat_model=chat_model)
    return await converter.convert(audio_desc, visual_desc, motion_desc="", shot_duration=shot_duration)


# 示例用法
if __name__ == "__main__":
    # 测试转换
    audio_desc = """
[Sound Effect] Ambient cafe music
Alice (Happy): 你好Bob，今天天气真不错！
Bob (Curious): 是啊，阳光明媚。我们去咖啡馆坐坐吧？
Alice: 好主意！我知道附近有一家很棒的咖啡馆。
"""
    
    visual_desc = """
Medium shot at eye level. A cozy cafe with warm lighting and wooden furniture. 
Alice is sitting by the window, smiling happily. Bob sits across from her, leaning forward with curiosity.
"""
    
    converter = PromptConverter()
    result = converter.rule_based_convert(audio_desc, visual_desc)
    print("Rule-based result:")
    print(result)
    print()
    
    # 解析测试
    parsed = converter.parse_audio_description(audio_desc)
    print("Parsed dialogues:")
    for d in parsed.dialogues:
        print(f"  - {d.speaker} ({d.emotion}): {d.dialogue}")
