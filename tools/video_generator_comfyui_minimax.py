"""
VideoGeneratorComfyUILTX: 使用 MiniMax H3 模型生成视频

通过 ComfyUI 工作流调用本地部署的 MiniMax H3 模型

Usage:
    # 方式1: 直接实例化
    generator = VideoGeneratorComfyUIMiniMax(
        base_url="http://127.0.0.1:8188"
    )
    
    # 多帧视频生成
    video = await generator.generate_single_video(
        prompt="A cinematic scene in a cafe. Alice smiles warmly, saying '你好'. Bob leans forward, curious.",
        reference_image_paths=["/path/to/frame0.png", "/path/to/frame1.png", "/path/to/frame2.png"],
    )
    
    # 方式2: 从配置实例化
    generator = VideoGeneratorComfyUIMiniMax.from_config(config)
"""


import logging
import uuid
import random
from typing import Any, List, Optional, Literal

from interfaces.video_output import VideoOutput
from tools.comfyui_workflow_runner import ComfyUIWorkflowRunner
from utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)

base_workflow_path = "workflows/MiniMax_base.json"
base_ui_workflow_path = "workflows/MiniMax_base_ui.json"
base_output_node_ids = ["92"]
ref_workflow_path = "workflows/MiniMax_ref.json"
ref_ui_workflow_path = "workflows/MiniMax_ref_ui.json"
ref_output_node_ids = ["92"]
max_noise = 2**50 - 1
aspect_ratio_map = {
    "1:1": '1:1 (Square)', 
    "2:3": '2:3 (Portrait Photo)', 
    "3:2": '3:2 (Photo)', 
    "3:4": '3:4 (Portrait Standard)', 
    "4:3": '4:3 (Standard)', 
    "9:16": '9:16 (Portrait Widescreen)', 
    "16:9": '16:9 (Widescreen)', 
    "21:9": '21:9 (Ultrawide)',
}

class VideoGeneratorComfyUIMiniMax:
    """
    MiniMax H3 视频生成器
    
    基于 ComfyUI 工作流，支持：
    - 首帧视频生成
    - 首尾帧视频生成
    - TODO:: 多参考视频生成（场景、角色一致性、工作台-like tv 设置路径）
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        初始化 MiniMax H3 视频生成器
        
        Args:
            base_url: ComfyUI 服务地址
            rate_limiter: 速率限制器
        """
        self.base_url = base_url
        self.rate_limiter = rate_limiter
    
    @classmethod
    def from_config(cls, config: dict[str, Any], rate_limiter: Optional[RateLimiter] = None) -> "VideoGeneratorComfyUIMiniMax":
        """从配置字典创建实例"""
        return cls(
            base_url=config.get("base_url", "http://127.0.0.1:8188"),
            rate_limiter=rate_limiter,
        )
        
    async def load_base_workflow(
        self, 
        runner: ComfyUIWorkflowRunner,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        aspect_ratio: str = "16:9",
        fps: Literal[16, 24] = 16,
        duration: Literal[5, 10] = 5,
        megapixels: float = 0.8,
        step: int = 10,
    ) -> dict[str, Any]:
        """
        加载首帧工作流
        """
        
        workflow = runner.load_workflow(base_workflow_path)
        first_frame_name = await runner.upload_image(reference_image_paths[0])
        workflow["114"]["inputs"]["image"] = first_frame_name
        condition_node = workflow["105:104"]["inputs"]
        condition_node["prompt"] = prompt

        if len(reference_image_paths) == 2:
            last_frame_name = await runner.upload_image(reference_image_paths[1])
            workflow["129"]["inputs"]["image"] = last_frame_name
        else:
            workflow.pop("129", None)
            condition_node.pop("last_frame", None)

        resolution_selector = workflow["115"]["inputs"]
        resolution_selector["aspect_ratio"] = aspect_ratio_map[aspect_ratio]
        resolution_selector["megapixels"] = megapixels

        noise_seed = random.randint(1, max_noise)
        
        workflow["92"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["105:15"]["inputs"]["noise_seed"] = noise_seed
        workflow["105:111"]["inputs"]["value"] = duration
        workflow["105:9"]["inputs"]["value"] = step
            
        # ui
        ui_workflow = runner.load_workflow(base_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 92), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())
        first_frame_node = next((node for node in nodes if node["id"] == 114), None)
        first_frame_node["widgets_values"][0] = first_frame_name
        last_frame_node = next((node for node in nodes if node["id"] == 129), None)

        if len(reference_image_paths) == 2:
            last_frame_node["widgets_values"][0] = last_frame_name
        else:
            last_frame_node["mode"] = 4

        resolution_selector_node = next((node for node in nodes if node["id"] == 115), None)
        resolution_selector_node["widgets_values"][0] = aspect_ratio_map[aspect_ratio]
        resolution_selector_node["widgets_values"][1] = megapixels

        video_node = next((node for node in nodes if node["id"] == 105), None)
        video_node["widgets_values"][0] = prompt
        video_node["widgets_values"][4] = noise_seed
        video_node["widgets_values"][3] = duration

        subgraph_nodes = ui_workflow["definitions"]["subgraphs"][0]["nodes"]
        step_node = next((node for node in subgraph_nodes if node["id"] == 9), None)
        step_node["widgets_values"][1] = step
        
        return workflow, ui_workflow
    
    async def load_ref_workflow(
        self, 
        runner: ComfyUIWorkflowRunner,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        aspect_ratio: str = "16:9",
        fps: Literal[16, 24] = 16,
        duration: Literal[5, 10] = 5,
        megapixels: float = 0.8,
        step: int = 10,
    ) -> dict[str, Any]:
        """
        加载参考工作流
        """
        
        workflow = runner.load_workflow(ref_workflow_path)
        workflow["138"]["inputs"]["value"] = first_frame_name = prompt
        condition_node = workflow["136"]["inputs"]

        resolution_selector = workflow["115"]["inputs"]
        resolution_selector["aspect_ratio"] = aspect_ratio_map[aspect_ratio]
        resolution_selector["megapixels"] = megapixels

        noise_seed = random.randint(1, max_noise)
        
        workflow["92"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["129"]["inputs"]["noise_seed"] = noise_seed
        workflow["132"]["inputs"]["value"] = duration
        workflow["124"]["inputs"]["steps"] = step
            
        # ui
        ui_workflow = runner.load_workflow(ref_ui_workflow_path)
        nodes = ui_workflow["nodes"]

        prompt_node = next((node for node in nodes if node["id"] == 138), None)
        prompt_node["widgets_values"][0] = prompt

        resolution_selector_node = next((node for node in nodes if node["id"] == 115), None)
        resolution_selector_node["widgets_values"][0] = aspect_ratio_map[aspect_ratio]
        resolution_selector_node["widgets_values"][1] = megapixels

        output_node = next((node for node in nodes if node["id"] == 92), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())

        step_node = next((node for node in nodes if node["id"] == 124), None)
        step_node["widgets_values"][1] = step

        duration_node = next((node for node in nodes if node["id"] == 132), None)
        duration_node["widgets_values"][0] = duration

        ui_condition_node = next((node for node in nodes if node["id"] == 136), None)
        condition_node_inputs = ui_condition_node["inputs"]

        # 图片上传
        image_node_ids = [137, 139, 141, 142, 143]
        image_len = len(reference_image_paths)

        for i in range(image_len):
            image_path = await runner.upload_image(reference_image_paths[i])
            image_node_id = image_node_ids[i]

            workflow[str(image_node_id)]["inputs"]["image"] = image_path

            image_node = next((node for node in nodes if node["id"] == image_node_id), None)
            image_node["widgets_values"][0] = image_path

        # 移除无图片节点
        for i in range(image_len, 5):
            image_node_id = image_node_ids[i]

            workflow.pop(str(image_node_id), None)
            condition_node.pop(f"ref_images.ref_image_{i}", None)

            image_node = next((node for node in nodes if node["id"] == image_node_id), None)
            image_node["mode"] = 4
            image_link = next((link for link in condition_node_inputs if link["label"] == f"ref_images.ref_image_{i}"), None)
            image_link["link"] = None

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
        enable_schedule_mode: Optional[bool] = False,
        workflow_name: Optional[str] = 'base',
    ) -> VideoOutput | None:
        """
        生成单个视频
        
        Args:
            prompt: 视频描述文本（MiniMax 格式，英文+中文对话）
            reference_image_paths: 参考图片路径列表（兼容旧接口）
            audio_path: 音频路径
            resolution: 视频分辨率
            aspect_ratio: 视频宽高比
            fps: 视频帧率
            duration: 视频时长
        
        Returns:
            VideoOutput: 生成的视频输出
        
        示例:
            # 首帧
            video = await generator.generate_single_video(
                prompt="A cinematic scene in a cafe.",
                reference_image_paths=["/path/to/first_frame.png"]
            )
            
            # 首尾帧
            video = await generator.generate_single_video(
                prompt="A cinematic scene in a cafe. Alice says '你好'.",
                reference_image_paths=[
                    "/path/to/first_frame.png",
                    "/path/to/last_frame.png",
                ]
            )

            # TODO:: ref
        """
        logger.info("Generating video with MiniMax H3...")
        
        runner = ComfyUIWorkflowRunner(
            base_url=self.base_url,
            rate_limiter=self.rate_limiter,
        )
        workflow = {}
        audio_name = None
        
        # if audio_path:
            # audio_name = await runner.upload_audio(audio_path)
            
        
        if len(reference_image_paths) > 5:
            logger.warning("Too many reference images, only the first 9 will be used")
            reference_image_paths = reference_image_paths[:5]
        
        len_reference_image_paths = len(reference_image_paths)
        
        if workflow_name == 'base':
            logger.info("============Using MiniMax base workflow============")
            workflow, ui_workflow = await self.load_base_workflow(
                runner=runner,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fps=fps,
                duration=duration,
            )
            workflow_path = base_workflow_path
            output_node_ids = base_output_node_ids
        else:
            logger.info("============Using MiniMax ref workflow============")
            workflow, ui_workflow = await self.load_ref_workflow(
                runner=runner,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fps=fps,
                duration=duration,
            )
            workflow_path = ref_workflow_path
            output_node_ids = ref_output_node_ids

        ready = await runner.prepare(
            workflow_path=workflow_path,
            workflow=workflow,
            output_node_ids=output_node_ids,
            ui_workflow=ui_workflow,
            enable_schedule_mode=enable_schedule_mode,
            type="video",
        )

        if not ready:
            return

        # 执行工作流
        outputs = await runner.run(
            workflow_path=workflow_path,
            workflow=workflow,
            output_node_ids=output_node_ids,
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
ComfyUIVideoGenerator = VideoGeneratorComfyUIMiniMax
