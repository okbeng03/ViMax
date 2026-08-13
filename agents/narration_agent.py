from typing import List, Optional
import json
import asyncio
import os
import subprocess
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from utils.retry import after_func
from utils.completion_logger import log_agent

from langchain.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import create_chat_model
from configs.config import model_name

system_prompt_template_narration = \
"""
[Role]
你是一位专业的影视旁白编剧与节奏控制专家。你擅长为短视频场景撰写高度精炼的旁白，并严格根据镜头时长和对话间隙来编排旁白的出现时机，确保旁白、对话、画面三者和谐交融。

[Task]
根据提供的场景故事描述和分镜信息，为每个镜头生成一段简洁的旁白，并给出精确的旁白起始时间。你的旁白必须：
1. 在给定的“旁白可用时间窗口”内完整播放，不得超出。
2. 与已存在的对话音频保留至少0.5秒的静音间隙，绝不重叠。
3. 用简练的词句点明画面未尽的故事信息、情绪或转场逻辑，提升整体叙事连贯性。

[Input]
你将收到
- 场景故事描述: 整个场景的故事描述，分镜信息是基于它拆分的。包含在<story></story>
- 分镜信息: 故事板镜头列表。包含在<storyboard></storyboard>
故事板列表，你将收到一个 JSON 对象，包含多个镜头（shot）的信息。每个镜头的结构如下：
{shot_schema}
整个输入将是一个数组，如 [shot_01, shot_02, ...]。
- 用户需求（可选）：用户可以指定的可选约束或指导。包含在<USER_REQUIREMENT>和</USER_REQUIREMENT>。例如
  -目标受众：例如，儿童（7-12岁），年轻人，成年人，所有年龄段。 
  -故事类型/类型：例如，科幻，幻想，神秘，浪漫，喜剧，悲剧，现实主义，短片，电影剧本概念。 
  -长度：例如，5个关键场景，一个紧凑的故事适合10分钟的短片。 
  -其他：例如，需要一个扭曲的结局，关于爱和牺牲的主题，包括一段引人注目的对话。

[Output]
{format_instructions}

[Guidelines]
1. **时长严格控制**  
   - 中文旁白语速按约 4.5 字/秒 估算（含自然停顿）。  
   - 生成旁白后，请估算字数并计算预估时长 = 字数 / 4.5。  
   - 该时长必须 ≤ 所选 available_interval 的可用时长（end - start）。  
   - 如果时间不够，宁可缩短旁白或直接放弃该镜头，绝不能超出。

2. **避免与对话重叠**  
   - 只使用 available_intervals 中非零区间放置旁白。  
   - 旁白 start_time 必须在某个 interval 内，且 start_time + estimated_duration ≤ interval.end。  
   - 若 dialogue_intervals 为空，则整个镜头时长都可作为可用区间（但建议仍保留开头0.2秒和结尾0.3秒的呼吸间隙）。

3. **内容创作准则**  
   - **补充而非重复**：旁白不说画面上正在发生的动作，而是揭示背景、内心活动、过渡逻辑或情感潜台词。  
   - **简短有力**：每句旁白不要超过镜头长度。  
   - **保持连贯**：阅读下一个场景，可在当前镜头末尾用旁白引导下一场景，形成“声先于画”的桥梁。  
   - **口吻统一**：所有旁白采用同一叙述者身份（如客观第三人称温和叙述），避免突然改变视角。

4. **静默优先原则**  
   - 当可用时间不足以容纳任何有意义的旁白（如 < 1.0 秒），不要强行填入词语，而是留白。有时让画面自己说话更有力量。

5. **边界情况处理**  
   - 如果某 shot 完全没有对话，且 storyboard 信息量已足够，可不加旁白。  
   - 如果连续多个 shot 都很短且共用一段旁白更自然，可以将旁白放在第一个 shot 的尾部或跨越两个 shot（此时需要在输出中注明或拆分为两个条目分别标 start_time），但为了流水线简洁，建议每个 shot 独立生成。

请严格遵循以上规则，直接输出 JSON，不要添加解释。
"""

human_prompt_template_narration = \
"""
<story>
{story}
</story>

<storyboard>
{storyboard}
</storyboard>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""

class NarrationItem(BaseModel):
    idx: int = Field(..., description="镜头索引，从0开始递增")
    narration_text: Optional[str] = Field(default=None, description="旁白文本")
    start_time: Optional[float] = Field(default=0, description="旁白开始时间（秒），以镜头内部时间为基准")
    estimated_duration: Optional[float] = Field(default=0, description="旁白预估时长（秒），用于后续TTS校验")
    shot_duration: float = Field(default=0, description="镜头总时长（秒）")

class Interval(BaseModel):
    start: float = Field(..., description="时间区间开始时间（秒）")
    end: float = Field(..., description="时间区间结束时间（秒）")

class ShotItem(BaseModel):
    idx: str = Field(..., description="镜头索引，从0开始递增")
    shot_duration: float = Field(..., description="镜头总时长（秒）")
    dialogue_intervals: Optional[List[Interval]] = Field(default=[], description="对话占用的时间区间（秒）")
    available_intervals: Optional[List[Interval]] = Field(default=[], description="旁白可以使用的安全时间区间（已扣除对话及0.5秒安全间隙）")
    visual_desc: str = Field(..., description="镜头的视觉描述")
    audio_desc: str = Field(..., description="镜头的音频描述。可能包含[Sound Effect]、[Music]和[Speaker]")


class NarrationAgent:
    def __init__(
            self,
            chat_model: BaseChatModel,
            audio_generator
        ):

        self.chat_model = create_chat_model(
            model_provider="qwen",
            model=model_name.get("secondary", "deepseek-v4-flash-0731"),
        )
        self.audio_generator = audio_generator

    @log_agent("NarrationAgent")
    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_narration(self, story: str, storyboard: List[ShotItem], user_requirement: str, retry_timeout: int = 300) -> List[NarrationItem]:
        class NarrationResponse(BaseModel):
            narration: List[NarrationItem] = Field(..., description="旁白列表")
            
        user_requirement_str = user_requirement.strip() if user_requirement else ""
        parser = PydanticOutputParser(pydantic_object=NarrationResponse)
        messages = [
            ('system', system_prompt_template_narration.format(format_instructions=parser.get_format_instructions(), shot_schema=ShotItem.model_json_schema())),
            ('user', human_prompt_template_narration.format(story=story, storyboard=json.dumps([shot.model_dump() for shot in storyboard], ensure_ascii=False, indent=4), user_requirement_str=user_requirement_str))
        ]
        chain = self.chat_model | parser
        response: NarrationResponse = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        narration = response.narration
        
        return narration
    
    async def generate_audio(self, narration: List[NarrationItem], output_path: str, comfyui_enable: bool = False):
        """
        生成旁白音频
        
        Args:
            narration: 旁白列表，包含每个镜头的旁白文本和时间信息
            output_path: 最终合成音频的输出路径
            working_dir: 工作目录，用于保存临时音频文件
            
        Returns:
            str: 音频文件路径
        """
        
        if not narration:
            # 如果没有旁白，创建一个空音频
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", "0.1", "-y", output_path
            ], check=True, capture_output=True)
            return output_path
        
        cache_dir = os.path.join(os.path.dirname(output_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 依次生成旁白音频
        # 合成音频到最终输出
        # 使用 concat 拼接音频，而非堆叠混合
        # 按时间顺序拼接：静音片段 + 旁白音频 + 静音片段
        concat_parts = []
        for i, item in enumerate(narration):
            if item.narration_text:
                # 前置静音
                if item.start_time is not None and item.start_time > 0:
                    silence_before = os.path.join(cache_dir, f"silence_before_{i}.flac")
                    subprocess.run([
                        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", str(item.start_time), "-y", silence_before
                    ], check=True, capture_output=True)
                    concat_parts.append(silence_before)
                    
                adjusted_path = os.path.join(cache_dir, f"narration_adjusted_{i}.flac")
                temp_path = os.path.join(cache_dir, f"narration_temp_{i}.flac")
            
                if os.path.exists(adjusted_path):
                    concat_parts.append(adjusted_path)
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", adjusted_path
                    ]
                    actual_duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())
                    item.estimated_duration = actual_duration
                else:
                    if not os.path.exists(temp_path):
                        if not comfyui_enable:
                            return

                        audio_output = await self.audio_generator.generate_single_audio(
                            prompt=item.narration_text,
                            character="旁白"
                        )
                        audio_output.save(temp_path)
                    
                    # 使用 ffmpeg 使音频时长等于 estimated_duration
                    # 获取音频实际时长
                    estimated_dur = item.estimated_duration or 0
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", temp_path
                    ]
                    actual_duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())
                    
                    if estimated_dur > 0 and abs(actual_duration - estimated_dur) > 0.1:
                        if actual_duration > estimated_dur:
                            # 音频过长，如果可以，调整预估时长，避免加速失真
                            remaining_duration = item.shot_duration - estimated_dur - (item.start_time or 0)
                            
                            if remaining_duration > 0.3:
                                estimated_dur = min(actual_duration, (item.shot_duration - (item.start_time or 0) - 0.3))
                                item.estimated_duration = estimated_dur
                        
                        # 计算播放速度：目标时长/实际时长
                        speed = actual_duration / estimated_dur
                        subprocess.run([
                            "ffmpeg", "-y", "-i", temp_path,
                            "-filter:a", f"atempo={speed}",
                            adjusted_path
                        ], check=True, capture_output=True)
                        temp_path = adjusted_path
                    
                    concat_parts.append(temp_path)

            # 后置静音（填充到总时长）
            silence_after = os.path.join(cache_dir, f"silence_after_{i}.flac")
            silence_after_duration = max(0.0, item.shot_duration - (item.start_time or 0) - (item.estimated_duration or 0))
            if silence_after_duration > 0:
                subprocess.run([
                    "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", str(silence_after_duration), "-y", silence_after
                ], check=True, capture_output=True)
                concat_parts.append(silence_after)
            
        if not concat_parts:
            # 如果没有生成任何音频，创建空音频
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", "0.1", "-y", output_path
            ], check=True, capture_output=True)
            return output_path

        # 使用concat拼接所有片段
        concat_inputs = []
        for path in concat_parts:
            concat_inputs.extend(["-i", path])
        
        filter_complex = f"concat=n={len(concat_parts)}:v=0:a=1[out]"

        subprocess.run([
            "ffmpeg", "-y",
            *concat_inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "44100",
            output_path
        ], check=True, capture_output=True)
        
        # 清理临时静音文件
        for path in concat_parts:
            if path.endswith("_before.flac") or path.endswith("_after.flac"):
                if os.path.exists(path):
                    os.remove(path)
