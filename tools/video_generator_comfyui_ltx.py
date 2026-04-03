"""
VideoGeneratorComfyUILTX: 使用 LTX Video 2.3 模型生成视频

通过 ComfyUI 工作流调用本地部署的 LTX Video 2.3 模型

支持的 ComfyUI 工作流格式:
1. 新格式 (nodes 数组):
   {
       "id": "...",
       "nodes": [{"id": 110, ...}, {"id": 111, ...}, ...]
   }
2. 旧格式 (直接节点):
   {"110": {...}, "111": {...}, ...}

LTX 2.3 特性：
- 支持多帧输入（6+ 帧）用于角色/场景一致性
- 英文提示词效果最佳
- 不支持对话格式（:），需要将对话转为场景描述

Usage:
    # 方式1: 直接实例化
    generator = VideoGeneratorComfyUILTX(
        base_url="http://127.0.0.1:8188",
        workflow_json_path="workflows/ltx2_3.json",
        input_node_descriptions=[
            # 文本提示词
            {"node_id": 110, "field": "prompt", "source": "prompt"},
            # 多帧输入（支持 6+ 帧）
            {"node_id": 111, "field": "image1", "source": "frame", "index": 0},
            {"node_id": 112, "field": "image2", "source": "frame", "index": 1},
            {"node_id": 113, "field": "image3", "source": "frame", "index": 2},
            {"node_id": 114, "field": "image4", "source": "frame", "index": 3},
            {"node_id": 115, "field": "image5", "source": "frame", "index": 4},
            {"node_id": 116, "field": "image6", "source": "frame", "index": 5},
        ]
    )
    
    # 多帧视频生成
    video = await generator.generate_single_video(
        prompt="A cinematic scene in a cafe. Alice smiles warmly, saying '你好'. Bob leans forward, curious.",
        frame_paths=["/path/to/frame0.png", "/path/to/frame1.png", "/path/to/frame2.png"],
    )
    
    # 方式2: 从配置实例化
    generator = VideoGeneratorComfyUILTX.from_config(config)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tenacity import retry, stop_after_attempt

from interfaces.video_output import VideoOutput
from tools.comfyui_workflow_runner import ComfyUIWorkflowRunner
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


class VideoGeneratorComfyUILTX:
    """
    LTX Video 2.3 视频生成器
    
    基于 ComfyUI 工作流，支持：
    - 文本到视频生成
    - 多帧输入（6+ 帧）用于关键帧插值和角色一致性
    - 关键帧时间点控制
    
    重要提示：
    - LTX 2.3 对英文提示词效果最佳
    - 不支持对话格式（如 "Alice: Hello"），请使用场景描述格式
    - 推荐格式：A cinematic scene. Alice smiles warmly, saying "你好". Bob nods.
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflow_json_path: str = "workflows/ltx2_3.json",
        input_node_descriptions: Optional[List[Dict[str, Any]]] = None,
        max_frames: int = 6,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        初始化 LTX 视频生成器
        
        Args:
            base_url: ComfyUI 服务地址
            workflow_json_path: LTX 工作流 JSON 文件路径
            input_node_descriptions: 输入节点描述列表
            max_frames: 最大支持的帧数
            rate_limiter: 速率限制器
        """
        # 默认的 LTX 工作流输入节点配置（使用整数节点 ID，支持 6 帧）
        default_input_descriptions = input_node_descriptions or [
            # 文本提示词
            {"node_id": 110, "field": "prompt", "source": "prompt"},
            # 多帧输入（支持 6+ 帧）
            {"node_id": 111, "field": "image1", "source": "frame", "index": 0},
            {"node_id": 112, "field": "image2", "source": "frame", "index": 1},
            {"node_id": 113, "field": "image3", "source": "frame", "index": 2},
            {"node_id": 114, "field": "image4", "source": "frame", "index": 3},
            {"node_id": 115, "field": "image5", "source": "frame", "index": 4},
            {"node_id": 116, "field": "image6", "source": "frame", "index": 5},
        ]
        
        self.runner = ComfyUIWorkflowRunner(
            base_url=base_url,
            workflow_json_path=workflow_json_path,
            input_node_descriptions=default_input_descriptions,
            output_node_ids=[],  # 自动检测
            rate_limiter=rate_limiter,
        )
        
        self.max_frames = max_frames
        logger.info(f"Initialized VideoGeneratorComfyUILTX with workflow: {workflow_json_path}, max_frames: {max_frames}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], rate_limiter: Optional[RateLimiter] = None) -> "VideoGeneratorComfyUILTX":
        """从配置字典创建实例"""
        return cls(
            base_url=config.get("base_url", "http://127.0.0.1:8188"),
            workflow_json_path=config.get("workflow_json_path", "workflows/ltx2_3.json"),
            input_node_descriptions=config.get("input_node_descriptions"),
            max_frames=config.get("max_frames", 6),
            rate_limiter=rate_limiter,
        )
    
    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        frame_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> VideoOutput:
        """
        生成单个视频
        
        Args:
            prompt: 视频描述文本（LTX 格式，英文+中文对话）
            reference_image_paths: 参考图片路径列表（兼容旧接口）
            frame_paths: 关键帧路径列表（按时间顺序，0=首帧）
            **kwargs: 其他参数
        
        Returns:
            VideoOutput: 生成的视频输出
        
        示例:
            # 单帧（首帧）
            video = await generator.generate_single_video(
                prompt="A cinematic scene in a cafe.",
                frame_paths=["/path/to/first_frame.png"]
            )
            
            # 多帧（首尾帧 + 中间帧）
            video = await generator.generate_single_video(
                prompt="A cinematic scene in a cafe. Alice says '你好'.",
                frame_paths=[
                    "/path/to/frame_0s.png",   # 0s
                    "/path/to/frame_2s.png",   # 2s - Alice enters
                    "/path/to/frame_5s.png",   # 5s - last frame
                ]
            )
        """
        logger.info("Generating video with LTX 2.3...")
        
        # 兼容处理：reference_image_paths 可以作为 frame_paths 的别名
        if frame_paths is None and reference_image_paths:
            frame_paths = reference_image_paths
        
        # 限制帧数
        if frame_paths and len(frame_paths) > self.max_frames:
            logger.warning(f"Frame count {len(frame_paths)} exceeds max {self.max_frames}, truncating")
            frame_paths = frame_paths[:self.max_frames]
        
        # 执行工作流
        outputs = await self.runner.run(
            prompt=prompt,
            frames=frame_paths,
            reference_image_paths=reference_image_paths,
            **kwargs,
        )
        
        # 提取输出路径
        output_paths = self.runner.get_output_paths(outputs)
        
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
            self.generate_single_video(prompt=prompt, frame_paths=frames, **kwargs)
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
