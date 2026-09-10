import logging
import asyncio
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from interfaces import CharacterInScene
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import create_chat_model
from utils.completion_logger import log_agent
from configs.config import model_name
from interfaces.shot_description import ShotDescription
from agents.minimax.storyboard import StoryboardShot
from .base_en import base_system_prompt, base_user_prompt
from .ref_en import ref_system_prompt, ref_user_prompt

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# 后处理：订正 LLM 输出中被翻译成英文的对话
# ---------------------------------------------------------------------------
_ENGLISH_DIALOGUE_RE = re.compile(r"<d>\s*\[English\](.*?)</d>", re.DOTALL | re.IGNORECASE)
_DIALOGUE_LINE_RE = re.compile(r"^\s*\[(?:Speaker|对话|Dialogue)\]", re.IGNORECASE)
_QUOTED_TEXT_RE = re.compile(r'[“"](.+)[”"]\s*$')
_COLON_TEXT_RE = re.compile(r"[：:]\s*(.*)$")


def _extract_original_dialogues(audio_desc: str) -> List[str]:
    """从 audio_desc 中按出现顺序提取原始对白文本（仅处理 [Speaker]/[对话]/[Dialogue] 行）。"""
    dialogues: List[str] = []
    if not audio_desc:
        return dialogues
    for line in audio_desc.splitlines():
        line = line.strip()
        if not line or not _DIALOGUE_LINE_RE.match(line):
            continue
        # 优先取引号内的完整内容（内部可包含中文引号“”）
        m = _QUOTED_TEXT_RE.search(line)
        if m:
            dialogues.append(m.group(1).strip())
            continue
        # 兜底：无引号时取冒号后的内容
        m = _COLON_TEXT_RE.search(line)
        if m:
            dialogues.append(m.group(1).strip())
    return dialogues


def _fix_english_dialogues(prompt: str, audio_desc: str) -> str:
    """
    检查 LLM 输出中的 <d>[English] ...</d> 英文对话；
    若命中，则从 audio_desc 提取原始中文对白，按出现顺序替换为 <d>[Chinese] 原文</d>。
    英文对话数量多于原始对白时，多余部分保持原样。
    """
    if not prompt or not audio_desc:
        return prompt
    if not _ENGLISH_DIALOGUE_RE.search(prompt):
        return prompt
    original_dialogues = _extract_original_dialogues(audio_desc)
    if not original_dialogues:
        return prompt

    replaced = 0

    def _replacer(match: re.Match) -> str:
        nonlocal replaced
        if replaced >= len(original_dialogues):
            return match.group(0)
        original = original_dialogues[replaced]
        replaced += 1
        return f"<d>[Chinese] {original}</d>"

    fixed_prompt = _ENGLISH_DIALOGUE_RE.sub(_replacer, prompt)
    if replaced:
        logger.warning("检测到 %d 处英文对话，已用 audio_desc 中的原始中文对白订正", replaced)
    return fixed_prompt


class PromptConverter:
    """
    Prompt 转换器
    
    负责将对话格式的 prompt 转换为适合 minimax h3 的场景描述格式
    """
    
    def __init__(
        self,
    ):
        self.chat_model = create_chat_model(
            model_provider="qwen",
            model=model_name.get("tertiary", "deepseek-v4-flash-0731"),
        )

    @log_agent("MiniMaxPromptConverter")
    async def convert(
        self,
        style: str,
        shot_description: ShotDescription,
        characters: List[CharacterInScene] = [],
        retry_timeout: int = 300,
    ) -> ShotDescriptionWithDialogues:
        """
        转换 prompt 为 minimax h3 base 格式
        
        Args:
            style: 风格
            shot_description: 镜头描述
            characters: 角色列表
        """
        parser = PydanticOutputParser(pydantic_object=ShotDescriptionWithDialogues)

        # 从 shot_description ff_vis_char_idxs 和 lf_vis_char_idxs 中过滤包含在镜头中的角色。并集 + 唯一
        vis_char_idxs = []
        seen = set()
        for idx in [*shot_description.ff_vis_char_idxs, *shot_description.lf_vis_char_idxs]:
            if idx not in seen:
                seen.add(idx)
                vis_char_idxs.append(idx)
        characters = [characters[idx] for idx in vis_char_idxs]

        characters_str = "\n".join([f"{character.identifier_in_scene}: {character.static_features}{character.dynamic_features}" for character in characters])
        task_type = "I2VA"
        frame_description = f"first frame description: {shot_description.ff_desc}"

        if shot_description.variation_type in ["medium", "large"]:
            task_type = "FL2VA"
            frame_description += f"\nlast frame description: {shot_description.lf_desc}"

        messages = [
            SystemMessage(content=base_system_prompt),
            HumanMessage(content=base_user_prompt.format(
                task_type=task_type,
                style=style,
                frame_description=frame_description,
                motion_description=shot_description.motion_desc or "(No motion description)",
                audio_description=shot_description.audio_desc,
                shot_duration=shot_description.shot_duration,
                character_list=characters_str
            )),
        ]
        
        chain = self.chat_model
        result = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )

        prompt = _fix_english_dialogues(result.content, shot_description.audio_desc)

        return ShotDescriptionWithDialogues(prompt=prompt, shot_duration=shot_description.shot_duration)
            
    @log_agent("MiniMaxPromptBaseConverter")
    async def base_convert(
        self,
        style: str,
        shot_description: StoryboardShot,
        characters: List[CharacterInScene] = [],
        retry_timeout: int = 300,
    ) -> ShotDescriptionWithDialogues:
        """
        转换 prompt 为 minimax h3 base 格式
        
        Args:
            style: 风格
            shot_description: 镜头描述
            characters: 角色列表
        """
        parser = PydanticOutputParser(pydantic_object=ShotDescriptionWithDialogues)
        characters = [characters[idx] for idx in shot_description.characters]

        characters_str = "\n".join([f"{character.identifier_in_scene}: {character.static_features}{character.dynamic_features}" for character in characters])
        task_type = "FL2VA"
        frame_description = f"first frame description: {shot_description.first_frame_description}"
        frame_description += f"\nlast frame description: {shot_description.final_frame_description}"

        messages = [
            SystemMessage(content=base_system_prompt),
            HumanMessage(content=base_user_prompt.format(
                task_type=task_type,
                style=style,
                frame_description=frame_description,
                motion_description=shot_description.motion_description or "(No motion description)",
                audio_description=shot_description.audio_desc,
                shot_duration=shot_description.duration,
                character_list=characters_str
            )),
        ]
        
        chain = self.chat_model
        result = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )

        prompt = _fix_english_dialogues(result.content, shot_description.audio_desc)

        return ShotDescriptionWithDialogues(prompt=prompt, shot_duration=shot_description.duration)

    @log_agent("MiniMaxRefPromptConverter")
    async def ref_convert(
        self,
        style: str,
        shot_description: StoryboardShot,
        characters: List[CharacterInScene] = [],
        retry_timeout: int = 300,
    ) -> ShotDescriptionWithDialogues:
        """
        转换 prompt 为 minimax h3 base 格式
        
        Args:
            style: 风格
            shot_description: 镜头描述
            characters: 角色列表
        """
        parser = PydanticOutputParser(pydantic_object=ShotDescriptionWithDialogues)

        # 从 shot_description ff_vis_char_idxs 和 lf_vis_char_idxs 中过滤包含在镜头中的角色。并集 + 唯一
        characters = [characters[idx] for idx in shot_description.characters]

        characters_str = "\n".join([f"{character.identifier_in_scene}: {character.static_features}{character.dynamic_features}" for character in characters])
        reference_list = "\n".join([f"<Subject {i+1}> {character.static_features}{character.dynamic_features}" for i, character in enumerate(characters)])
        frame_description = ""

        if shot_description.first_frame_description:
            frame_description = f"first frame description: {shot_description.first_frame_description}"
            reference_list += f"\n<Picture 1> {shot_description.first_frame_description}"

        messages = [
            SystemMessage(content=ref_system_prompt),
            HumanMessage(content=ref_user_prompt.format(
                style=style,
                frame_description=frame_description,
                motion_description=shot_description.motion_description or "(No motion description)",
                audio_description=shot_description.audio_desc,
                shot_duration=shot_description.duration,
                character_list=characters_str,
                reference_list=reference_list,
            )),
        ]
        
        chain = self.chat_model
        result = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )

        prompt = _fix_english_dialogues(result.content, shot_description.audio_desc)

        return ShotDescriptionWithDialogues(prompt=prompt, shot_duration=shot_description.duration)
            