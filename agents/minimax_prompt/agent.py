import logging
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from interfaces import CharacterInScene
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import create_chat_model
from utils.completion_logger import log_agent
from configs.config import model_name
from interfaces.shot_description import ShotDescription
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
    
        return ShotDescriptionWithDialogues(prompt=result.content, shot_duration=shot_description.shot_duration)
            
