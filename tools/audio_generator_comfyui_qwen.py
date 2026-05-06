"""
AudioGeneratorComfyUIQwenTTS: 使用 Qwen.tts 模型生成音频

通过 ComfyUI 工作流调用本地部署的 Qwen.tts 模型

Usage:
    # 方式1: 直接实例化
    generator = AudioGeneratorComfyUIQwenTTS(
        base_url="http://127.0.0.1:8188"
    )
    
    audio = await generator.generate_single_audio(
        prompt="A cute cat",
        character="Cat"
    )
    
    # 方式2: 从配置实例化
    generator = AudioGeneratorComfyUIQwenTTS.from_config(config)
"""

import logging
import uuid
import random
import os
import subprocess
import shutil
from typing import Any, List, Optional
from interfaces.audio_output import AudioOutput
from tools.comfyui_workflow_runner import ComfyUIWorkflowRunner
from utils.rate_limiter import RateLimiter
from agents.prompt_converter import Dialogue

logger = logging.getLogger(__name__)

workflow_path = "workflows/tts_qwen.json"
output_node_ids = ["64"]
max_noise = 2**50 - 1

class AudioGeneratorComfyUIQwenTTS:
    """
    Qwen.tts 音频生成器
    
    基于 ComfyUI 工作流，支持：
    - 预设音色生成音频
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        初始化 Flux 图片生成器
        
        Args:
            base_url: ComfyUI 服务地址
            rate_limiter: 速率限制器
        """
        
        self.base_url = base_url
        self.rate_limiter = rate_limiter
    
    @classmethod
    def from_config(cls, config: dict[str, Any], rate_limiter: Optional[RateLimiter] = None) -> "ImageGeneratorComfyUIFlux":
        """从配置字典创建实例"""
        return cls(
            base_url=config.get("base_url", "http://127.0.0.1:8188"),
            rate_limiter=rate_limiter,
        )
    
    
    async def load_workflow(self, runner: ComfyUIWorkflowRunner, prompt: str, character: str) -> dict[str, Any]:
        """加载文生图工作流"""
    
        workflow = runner.load_workflow(workflow_path)
        workflow["64"]["inputs"]["filename_prefix"] = f"audio/{str(uuid.uuid4())}"
        workflow["51"]["inputs"]["filename"] = f"{character}.wav"
        workflow["52"]["inputs"]["target_text"] = prompt
        workflow["52"]["inputs"]["seed"] = random.randint(1, max_noise)
        
        return workflow
    
    async def generate_single_audio(
        self,
        prompt: str,
        character: str,
        **kwargs,
    ) -> AudioOutput:
        """
        生成音频
        
        Args:
            prompt: 音频描述文本
            character: 音色
            **kwargs: 其他参数
            
        Returns:
            AudioOutput: 生成的音频输出
        """
        logger.info("Generating audio with Qwen.tts...")
        
        runner = ComfyUIWorkflowRunner(
            base_url=self.base_url,
            rate_limiter=self.rate_limiter,
        )
        workflow = await self.load_workflow(runner=runner, prompt=prompt, character=character)
  
        # 执行工作流
        outputs = await runner.run(
            workflow_path=workflow_path,
            workflow=workflow,
            output_node_ids=output_node_ids,
        )
        # 提取输出路径
        output_paths = runner.get_output_paths(outputs)
        
        if not output_paths:
            raise RuntimeError("No audio output generated")
        
        # 获取第一个输出节点的结果
        first_output = list(output_paths.values())[0]

        if isinstance(first_output, list) and len(first_output) > 0:
            # 输出是音频路径列表
            audio_path = first_output[0]
            return AudioOutput(fmt="url", ext="flac", data=audio_path)
        elif isinstance(first_output, str):
            # 输出是单个音频路径
            return AudioOutput(fmt="url", ext="flac", data=first_output)
        else:
            raise RuntimeError(f"Unexpected output format: {first_output}")

    async def adjust_audio(
        self,
        duration: float,
        dialogues: List[Dialogue],
        shot_path: str,
        **kwargs,
    ):
        """
        调整音频
        根据对话列表和视频时长，调整对话音频文件，保证视频时长与对话时长一致
        
        规则：
        1. 先计算每个对话的时长，订正对话列表的时长
        2. 按总时长不能大于视频时长 * 1.3判断，判断对话时长是否符合要求
           - 如果大于，则raise error，提示分镜
        3. 检查并调整dialogues item 的 start_time，确保最后的对话结束时间不超过总时长
           - 在满足不超过总时长的情况下，首先满足推荐的 start_time
           - 否则，调整
             - 如果可以，两个对话之间保留间隔至少0.5
        4. 计算 speed_up_ratio
        5. 如果 speed_up_ratio > 1, 对 dialogues 进行加速
        6. 最后，将对话拼接，添加开头、两个对话之间、结尾的间隙，保证对话总时长等于视频时长
        
        Args:
            duration: 视频时长
            dialogues: 对话列表
            shot_path: 对话音频保存路径。可以 f"{shot_path}/{idx}.flac" 获取对话音频
            **kwargs: 其他参数
        """
        
        # Step 1: 计算每个对话的实际时长，更新 dialogue.duration
        total_duration = 0.0
        for idx, dialogue in enumerate(dialogues):
            audio_path = os.path.join(shot_path, f"{idx}.flac")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            dialogue.duration = self._get_audio_duration(audio_path)
            total_duration += dialogue.duration
        
        logger.info(f"Total dialogue duration: {total_duration:.2f}s, video duration: {duration:.2f}s")
        
        # Step 2: 判断对话时长是否符合要求（不能大于视频时长 * 1.3）
        # max_allowed_duration = duration * 1.45
        # if total_duration > max_allowed_duration:
        #     raise ValueError(
        #         f"Dialogue total duration ({total_duration:.2f}s) exceeds maximum allowed "
        #         f"({max_allowed_duration:.2f}s). Please split this shot."
        #     )

        if total_duration <= duration:
            speed_ratio = 1.0
        else:
            speed_ratio = total_duration / duration
            logger.info(f"Applying speedup ratio: {speed_ratio:.4f}")

        # 如果 speed_up_ratio > 1, 调整 dialogues 的 duration
        if speed_ratio > 1.0:
            for dialogue in dialogues:
                dialogue.duration = dialogue.duration / speed_ratio
        
        # Step 3: 计算每个对话的 start_time 可接受范围，然后调整
        # 范围: [min_start, max_start]
        # - min_start: 考虑前一个对话 + 最小间隔
        # - max_start: 根据剩余时间计算，确保所有对话能放在视频时长内
        dialogues_len = len(dialogues)
        total_duration = sum([dialogue.duration for dialogue in dialogues])
        remaining_duration = duration - total_duration
        max_gap = 0 if dialogues_len == 1 else remaining_duration / (dialogues_len - 1)
        min_gap = max_gap if max_gap < 0.5 else 0.5
        remaining_gap = remaining_duration - min_gap * (dialogues_len - 1)
        
        # 计算每个对话的最小可能开始时间（基于前一个对话的 min_start）
        min_starts = [0.0]
        max_starts = [0.0 + remaining_gap]
        for i in range(1, dialogues_len):
            min_start = min_starts[i - 1] + dialogues[i - 1].duration + min_gap
            min_starts.append(min_start)
            max_start = min_start + remaining_gap
            max_starts.append(max_start)
        
        # 调整每个对话的 start_time
        for i, dialogue in enumerate(dialogues):
            original_start = dialogue.start_time
            min_start = min_starts[i]
            max_start = max_starts[i]
            
            if min_start <= original_start <= max_start:
                # 推荐时间在范围内，保持不变
                pass
            elif original_start < min_start:
                # 推荐时间太早，调整到最小值
                dialogue.start_time = min_start
                logger.debug(f"Dialogue {i}: start_time adjusted from {original_start:.2f}s to {min_start:.2f}s")
            else:
                # 推荐时间太晚，调整到最大值
                dialogue.start_time = max_start
                logger.debug(f"Dialogue {i}: start_time adjusted from {original_start:.2f}s to {max_start:.2f}s")
        
        # 校验最终时长
        if dialogues:
            last_dialogue = dialogues[-1]
            last_end_time = last_dialogue.start_time + last_dialogue.duration
            if last_end_time > duration:
                raise ValueError(
                    f"Dialogues ({last_end_time:.2f}s) exceed video duration ({duration:.2f}s). Please split this shot."
                )
        
        # Step 4: 处理并拼接所有对话音频
        temp_dir = os.path.join(shot_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            audio_paths: List[str] = []
            
            for idx, dialogue in enumerate(dialogues):
                audio_path = os.path.join(shot_path, f"{idx}.flac")
                adjusted_path = os.path.join(temp_dir, f"{idx}_adjusted.flac")
                
                if speed_ratio > 1.0:
                    self._speed_up_audio(audio_path, adjusted_path, speed_ratio)
                else:
                    subprocess.run(["cp", audio_path, adjusted_path], check=True)
                
                audio_paths.append(adjusted_path)

            final_audio_path = os.path.join(shot_path, "final_dialogues.flac")
            self._concatenate_audio(
                dialogues,
                audio_paths, 
                final_audio_path, 
                duration, 
            )
            
            return final_audio_path, dialogues
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频文件时长（秒）"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Failed to get audio duration: {e}")
    
    def _speed_up_audio(self, input_path: str, output_path: str, speed_ratio: float) -> None:
        """
        使用 ffmpeg 加速音频
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            speed_ratio: 加速比（>1 表示加速，<1 表示减速）
        """
        try:
            # 使用 atempo 滤镜进行速度调整
            # atempo 范围是 0.5-2.0，所以如果需要更大比例需要链式调用
            if speed_ratio > 2.0:
                # 分多次加速
                atempo1 = min(speed_ratio, 2.0)
                atempo2 = speed_ratio / atempo1
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_path,
                    "-filter:a", f"atempo={atempo1},atempo={atempo2}",
                    "-ar", "44100", output_path
                ], check=True, capture_output=True)
            elif speed_ratio < 0.5:
                # 减速
                atempo1 = max(speed_ratio, 0.5)
                atempo2 = speed_ratio / atempo1
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_path,
                    "-filter:a", f"atempo={atempo1},atempo={atempo2}",
                    "-ar", "44100", output_path
                ], check=True, capture_output=True)
            else:
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_path,
                    "-filter:a", f"atempo={speed_ratio}",
                    "-ar", "44100", output_path
                ], check=True, capture_output=True)
            
            logger.debug(f"Speed up audio: {input_path} -> {output_path} (ratio={speed_ratio:.2f})")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to speed up audio: {e}")
            raise RuntimeError(f"Failed to speed up audio: {e}")
    
    def _concatenate_audio(
        self, 
        dialogues: List[Dialogue],
        audio_segments: List[str],
        output_path: str, 
        target_duration: float
    ) -> None:
        """
        拼接音频并填充到目标时长
        
        Args:
            dialogues: 对话列表（包含 start_time 和 duration）
            audio_segments: 音频片段路径列表（与 dialogues 一一对应）
            output_path: 输出文件路径
            target_duration: 目标总时长
        """
        try:
            # 使用 ffmpeg filter_complex 实现时间线拼接
            # 每个音频用 adelay 延迟到对应时间点，然后用 amix 混合
            
            # 构建 ffmpeg 输入参数
            inputs = []
            for audio_path in audio_segments:
                inputs.extend(["-i", audio_path])
            
            # 构建 filter_complex 字符串
            # 格式: [0:a]adelay={ms}|{ms}[a0];[1:a]adelay={ms}|{ms}[a1];...[a0][a1]...[an]amix=inputs={n}:duration=longest[out]
            delay_filters = []
            for i in range(len(dialogues)):
                delay_ms = int(dialogues[i].start_time * 1000)
                delay_filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
            
            mix_inputs = "".join(f"[a{i}]" for i in range(len(dialogues)))
            filter_complex = f"{';'.join(delay_filters)};{mix_inputs}amix=inputs={len(dialogues)}:duration=longest[out]"
            
            # 执行 ffmpeg
            subprocess.run([
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-t", str(target_duration),
                output_path
            ], check=True, capture_output=True)
            
            logger.info(f"Audio concatenated: {target_duration}s: {output_path}")
            
        except subprocess.CalledProcessError as e:
            # 删除 output_path
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"Failed to concatenate audio: {e}")
            raise RuntimeError(f"Failed to concatenate audio: {e}")

        except subprocess.CalledProcessError as e:
            # 删除 output_path
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"Failed to concatenate audio: {e}")
            raise RuntimeError(f"Failed to concatenate audio: {e}")
