"""
VideoGeneratorComfyUILTX: 使用 LTX Video 2.3 模型生成视频

通过 ComfyUI 工作流调用本地部署的 LTX Video 2.3 模型

LTX 2.3 特性：
- 支持多帧输入（6+ 帧）用于角色/场景一致性
- 英文提示词效果最佳
- 不支持对话格式（:），需要将对话转为场景描述

Usage:
    # 方式1: 直接实例化
    generator = VideoGeneratorComfyUILTX(
        base_url="http://127.0.0.1:8188"
    )
    
    # 多帧视频生成
    video = await generator.generate_single_video(
        prompt="A cinematic scene in a cafe. Alice smiles warmly, saying '你好'. Bob leans forward, curious.",
        reference_image_paths=["/path/to/frame0.png", "/path/to/frame1.png", "/path/to/frame2.png"],
    )
    
    # 方式2: 从配置实例化
    generator = VideoGeneratorComfyUILTX.from_config(config)
"""


import logging
import uuid
import random
from typing import Any, List, Optional, Literal

from interfaces.video_output import VideoOutput
from tools.comfyui_workflow_runner import ComfyUIWorkflowRunner
from utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)

first_frame_workflow_path = "workflows/LTX2_3_first_frame_audio.json"
first_frame_ui_workflow_path = "workflows/LTX2_3_first_frame_audio_ui.json"
mutil_frame_workflow_path = "workflows/LTX2_3_mutil_frame_audio.json"
mutil_frame_ui_workflow_path = "workflows/LTX2_3_mutil_frame_audio_ui.json"
first_frame_output_node_ids = ["75"]
mutil_frame_output_node_ids = ["649"]
mutil_sigmas = {
    1: "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    2: "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.82, 0.68, 0.52, 0.38, 0.18, 0.0",
    3: "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.86, 0.82, 0.78, 0.72, 0.66, 0.58, 0.5, 0.42, 0.34, 0.24, 0.12, 0.0",
    4: "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375,0.881203,0.863321,0.841251,0.820089,0.655, 0.381875, 0.0",
}
max_noise = 2**50 - 1

class VideoGeneratorComfyUILTX:
    """
    LTX Video 2.3 视频生成器
    
    基于 ComfyUI 工作流，支持：
    - 首帧视频生成
    - 多帧视频生成（最多4帧）用于关键帧插值和角色一致性
    
    重要提示：
    - LTX 2.3 对英文提示词效果最佳
    - 不支持对话格式（如 "Alice: Hello"），请使用场景描述格式
    - 推荐格式：A cinematic scene. Alice smiles warmly, saying "你好". Bob nods.
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        初始化 LTX 视频生成器
        
        Args:
            base_url: ComfyUI 服务地址
            rate_limiter: 速率限制器
        """

        self.base_url = base_url
        self.rate_limiter = rate_limiter
    
    @classmethod
    def from_config(cls, config: dict[str, Any], rate_limiter: Optional[RateLimiter] = None) -> "VideoGeneratorComfyUILTX":
        """从配置字典创建实例"""
        return cls(
            base_url=config.get("base_url", "http://127.0.0.1:8188"),
            rate_limiter=rate_limiter,
        )
        
    async def load_first_frame_workflow(
        self, 
        runner: ComfyUIWorkflowRunner,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        aspect_ratio: str = "16:9",
        fps: Literal[16, 24] = 16,
        duration: Literal[5, 10] = 5,
        audio_name: Optional[str] = None,
        use_xianxia_lora: bool = False,
        is_small_people: bool = False,
    ) -> dict[str, Any]:
        """
        加载首帧工作流
        """
        
        workflow = runner.load_workflow(first_frame_workflow_path)
        image_name = await runner.upload_image(reference_image_paths[0])
        noise_seed = random.randint(1, max_noise)
        
        workflow["75"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["269"]["inputs"]["image"] = image_name
        # workflow["270"]["inputs"]["noise_seed"] = noise_seed
        workflow["271"]["inputs"]["noise_seed"] = noise_seed
        workflow["294"]["inputs"]["value"] = fps
        workflow["313"]["inputs"]["value"] = prompt
        workflow["314"]["inputs"]["value"] = duration
        workflow["316"]["inputs"]["aspect_ratio"] = aspect_ratio
        
        # 如果是生成场景切换视频，因为缺少角色参考，会生成仙侠风任务。去掉该lora
        # if prompt.startswith("Two shots. The transition between the shots is a cut to. The style of the two shots should be consistent."):
        if not use_xianxia_lora:
            workflow["338"]["inputs"]["model"] = ["279", 0]

        if is_small_people:
            workflow["316"]["inputs"]["scale_to_length"] = 2560
        
        if audio_name:
            # 替换音频
            workflow["318"]["inputs"]["audio"] = audio_name
        else:
            # 没有音频，直接连接采样输出的audio
            workflow["304"]["inputs"]["audio"] = ["291", 0]
            
        # ui
        ui_workflow = runner.load_workflow(first_frame_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 75), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())
        image_node = next((node for node in nodes if node["id"] == 269), None)
        image_node["widgets_values"][0] = image_name
        # workflow["270"]["inputs"]["noise_seed"] = noise_seed
        noise_node = next((node for node in nodes if node["id"] == 271), None)
        noise_node["widgets_values"][0] = noise_seed
        fps_node = next((node for node in nodes if node["id"] == 294), None)
        fps_node["widgets_values"][0] = fps
        positive_node = next((node for node in nodes if node["id"] == 313), None)
        positive_node["widgets_values"][0] = prompt
        duration_node = next((node for node in nodes if node["id"] == 314), None)
        duration_node["widgets_values"][0] = duration
        scale_node = next((node for node in nodes if node["id"] == 316), None)
        scale_node["widgets_values"][0] = aspect_ratio
        
        # 如果是生成场景切换视频，因为缺少角色参考，会生成仙侠风任务。去掉该lora
        # if prompt.startswith("Two shots. The transition between the shots is a cut to. The style of the two shots should be consistent."):
        if not use_xianxia_lora:
            xianxia_node = next((node for node in nodes if node["id"] == 337), None)
            xianxia_node["mode"] = 4

        if is_small_people:
            scale_node = next((node for node in nodes if node["id"] == 316), None)
            scale_node["widgets_values"][7] = 2560
        
        if audio_name:
            # 替换音频
            tts_node = next((node for node in nodes if node["id"] == 318), None)
            tts_node["widgets_values"][0] = audio_name
        # else:
        #     # 没有音频，直接连接采样输出的audio
        #     ui_workflow["304"]["inputs"]["audio"] = ["291", 0]
        # 改连接太麻烦了。暂时不处理
        
        return workflow, ui_workflow
    
    async def load_mutil_frame_workflow(
        self, 
        runner: ComfyUIWorkflowRunner,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        aspect_ratio: str = "16:9",
        fps: int = 16,
        duration: int = 5,
        audio_name: Optional[str] = None,
        use_xianxia_lora: bool = False,
        is_small_people: bool = False,
    ) -> dict[str, Any]:
        """
        加载多帧工作流
        """
        
        workflow = runner.load_workflow(mutil_frame_workflow_path)
        workflow["649"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        noise_seed = random.randint(1, max_noise)
        workflow["636"]["inputs"]["noise_seed"] = noise_seed
        # workflow["632"]["inputs"]["noise_seed"] = noise_seed
        workflow["625"]["inputs"]["value"] = fps
        workflow["627"]["inputs"]["value"] = float(fps)
        workflow["672"]["inputs"]["text"] = prompt
        workflow["673"]["inputs"]["value"] = duration
        workflow["700"]["inputs"]["aspect_ratio"] = aspect_ratio
        
        if not use_xianxia_lora:
            workflow["605"]["inputs"]["model"] = ["610", 0]
        
        if audio_name:
            # 替换音频
            workflow["712"]["inputs"]["audio"] = audio_name
        else:
            # 没有音频，直接连接采样输出的audio
            workflow["649"]["inputs"]["audio"] = ["648", 0]

        if is_small_people:
            workflow["700"]["inputs"]["scale_to_length"] = 2560
        
        # 图片处理
        first_frame = {
            "image": "699",
            "scale": "700",
            "preprocess": "696",
            "addguide": "611"
        }
        last_frame = {
            "image": "686",
            "scale": "685",
            "preprocess": "684",
            "addguide": "638"
        }
        middle_frames = [
            {
                "image": "694",
                "scale": "693",
                "preprocess": "692",
                "addguide": "674",
            },
            {
                "image": "690",
                "scale": "689",
                "preprocess": "688",
                "addguide": "675",
            },
        ]
        
        reference_images_len = len(reference_image_paths)
        workflow["633"]["inputs"]["sigmas"] = mutil_sigmas[reference_images_len]
        all_frames = [first_frame] + middle_frames[:reference_images_len - 2] + [last_frame]
        
        for i, image_path in enumerate(reference_image_paths):
            image_name = await runner.upload_image(image_path)
            frame = all_frames[i]
            workflow[frame["image"]]["inputs"]["image"] = image_name

        for group in middle_frames[reference_images_len - 2:]:
            for node_id in group.values():
                workflow.pop(node_id, None)

        # 调整尾帧的 scale
        last_frame_scale = last_frame["scale"]
        prev_frame_scale_node_id = all_frames[-2]["scale"]
        workflow[last_frame_scale]["inputs"]["width"] = [prev_frame_scale_node_id, 3]
        workflow[last_frame_scale]["inputs"]["height"] = [prev_frame_scale_node_id, 3]
        last_frame_addguide = last_frame["addguide"]
        prev_frame_addguide_node_id = all_frames[-2]["addguide"]
        workflow[last_frame_addguide]["inputs"]["positive"] = [prev_frame_addguide_node_id, 0]
        workflow[last_frame_addguide]["inputs"]["negative"] = [prev_frame_addguide_node_id, 1]
        workflow[last_frame_addguide]["inputs"]["latent"] = [prev_frame_addguide_node_id, 2]
        
        # ui
        ui_workflow = runner.load_workflow(mutil_frame_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 649), None)
        output_node["widgets_values"]["filename_prefix"] = str(uuid.uuid4())
        noise_node = next((node for node in nodes if node["id"] == 636), None)
        noise_node["widgets_values"][0] = noise_seed
        fps_node = next((node for node in nodes if node["id"] == 625), None)
        fps_node["widgets_values"][0] = fps
        fps_float_node = next((node for node in nodes if node["id"] == 627), None)
        fps_float_node["widgets_values"][0] = float(fps)
        positive_node = next((node for node in nodes if node["id"] == 672), None)
        positive_node["widgets_values"][0] = prompt
        duration_node = next((node for node in nodes if node["id"] == 673), None)
        duration_node["widgets_values"][0] = duration
        scale_node = next((node for node in nodes if node["id"] == 700), None)
        scale_node["widgets_values"][0] = aspect_ratio
        first_image_node = next((node for node in nodes if node["id"] == 699), None)
        first_image_node["widgets_values"][0] = workflow["699"]["inputs"]["image"]
        last_image_node = next((node for node in nodes if node["id"] == 686), None)
        last_image_node["widgets_values"][0] = workflow["686"]["inputs"]["image"]
        
        if audio_name:
            # 替换音频
            tts_node = next((node for node in nodes if node["id"] == 712), None)
            tts_node["widgets_values"][0] = audio_name
        
        if not use_xianxia_lora:
            xianxia_node = next((node for node in nodes if node["id"] == 725), None)
            xianxia_node["mode"] = 4

        if is_small_people:
            scale_node = next((node for node in nodes if node["id"] == 700), None)
            scale_node["widgets_values"][7] = 2560

        return workflow, ui_workflow
    
    
    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        *,
        audio_path: Optional[str] = None,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        aspect_ratio: str = "16:9",
        fps: int = 15,
        duration: int = 5,
        use_xianxia_lora: bool = False,
        is_small_people: bool = False,
    ) -> VideoOutput:
        """
        生成单个视频
        
        Args:
            prompt: 视频描述文本（LTX 格式，英文+中文对话）
            reference_image_paths: 参考图片路径列表（兼容旧接口）
            audio_path: 音频路径
            resolution: 视频分辨率
            aspect_ratio: 视频宽高比
            fps: 视频帧率
            duration: 视频时长
        
        Returns:
            VideoOutput: 生成的视频输出
        
        示例:
            # 单帧（首帧）
            video = await generator.generate_single_video(
                prompt="A cinematic scene in a cafe.",
                reference_image_paths=["/path/to/first_frame.png"]
            )
            
            # 多帧（首尾帧 + 中间帧）
            video = await generator.generate_single_video(
                prompt="A cinematic scene in a cafe. Alice says '你好'.",
                reference_image_paths=[
                    "/path/to/frame_0s.png",   # 0s
                    "/path/to/frame_2s.png",   # 2s - Alice enters
                    "/path/to/frame_5s.png",   # 5s - last frame
                ]
            )
        """
        logger.info("Generating video with LTX 2.3...")
        
        runner = ComfyUIWorkflowRunner(
            base_url=self.base_url,
            rate_limiter=self.rate_limiter,
        )
        workflow = {}
        audio_name = None
        
        if audio_path:
            audio_name = await runner.upload_audio(audio_path)
        
        if len(reference_image_paths) > 4:
            logger.warning("Too many reference images, only the first 4 will be used")
            reference_image_paths = reference_image_paths[:4]
        
        len_reference_image_paths = len(reference_image_paths)
        
        if len_reference_image_paths >= 2:
            logger.info("============Using mutil frame workflow============")
            workflow, ui_workflow = await self.load_mutil_frame_workflow(
                runner=runner,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fps=fps,
                duration=duration,
                audio_name=audio_name,
                use_xianxia_lora=use_xianxia_lora,
                is_small_people=is_small_people,
            )
        else:
            logger.info("============Using first frame workflow============")
            workflow, ui_workflow = await self.load_first_frame_workflow(
                runner=runner,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fps=fps,
                duration=duration,
                audio_name=audio_name,
                use_xianxia_lora=use_xianxia_lora,
                is_small_people=is_small_people,
            )

        # 执行工作流
        outputs = await runner.run(
            workflow_path=mutil_frame_workflow_path if len_reference_image_paths >= 2 else first_frame_workflow_path,
            workflow=workflow,
            output_node_ids=mutil_frame_output_node_ids if len_reference_image_paths >= 2 else first_frame_output_node_ids,
            ui_workflow=ui_workflow,
            timeout=3000,  # 10 分钟超时
        )
        
        # 提取输出路径
        output_paths = runner.get_output_paths(outputs)
        
        if not output_paths:
            raise RuntimeError("No video output generated")
        
        # 获取视频路径
        video_path = None
        for node_id, output in output_paths.items():
            if output:
                if isinstance(output, list) and len(output) > 0:
                    video_path = output[0]
                    break
                elif isinstance(output, str):
                    video_path = output
                    break
        
        if not video_path:
            raise RuntimeError("Failed to extract video path from output")
        
        logger.info(f"Video generated: {video_path}")
        
        return VideoOutput(fmt="url", ext="mp4", data=video_path)
    
    async def generate_videos(
        self,
        prompts: List[str],
        frame_paths_list: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> List[VideoOutput]:
        """
        批量生成视频（并行）
        
        Args:
            prompts: 视频描述文本列表
            frame_paths_list: 关键帧路径列表的列表
            **kwargs: 其他参数
        
        Returns:
            视频输出列表
        """
        import asyncio
        
        if frame_paths_list is None:
            frame_paths_list = [None] * len(prompts)
        
        tasks = [
            self.generate_single_video(prompt=prompt, reference_image_paths=frames, **kwargs)
            for prompt, frames in zip(prompts, frame_paths_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤错误
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate video {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results


# 为了保持向后兼容，也导出通用类
ComfyUIVideoGenerator = VideoGeneratorComfyUILTX
