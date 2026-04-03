"""
ImageGeneratorComfyUIFlux: 使用 Flux.klein 模型生成图片

通过 ComfyUI 工作流调用本地部署的 Flux.klein 模型

支持的 ComfyUI 工作流格式:
1. 新格式 (nodes 数组):
   {
       "id": "...",
       "nodes": [{"id": 103, ...}, ...]
   }
2. 旧格式 (直接节点):
   {"103": {...}, "105": {...}, ...}

Usage:
    # 方式1: 直接实例化
    generator = ImageGeneratorComfyUIFlux(
        base_url="http://127.0.0.1:8188",
        workflow_json_path="workflows/flux_klein.json",
        input_node_descriptions=[
            {"node_id": 103, "field": "text_g", "source": "prompt"},
            {"node_id": 104, "field": "text_l", "source": "prompt"},
            {"node_id": 105, "field": "image", "source": "reference_image", "max_count": 4},
        ]
    )
    
    image = await generator.generate_single_image(
        prompt="A cute cat",
        reference_image_paths=["/path/to/style.png"]
    )
    
    # 方式2: 从配置实例化
    generator = ImageGeneratorComfyUIFlux.from_config(config)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tenacity import retry, stop_after_attempt

from interfaces.image_output import ImageOutput
from tools.comfyui_workflow_runner import ComfyUIWorkflowRunner
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


class ImageGeneratorComfyUIFlux:
    """
    Flux.klein 图片生成器
    
    基于 ComfyUI 工作流，支持：
    - 文本到图片生成
    - 参考图片输入（支持多张）
    - IP-Adapter 等控制网络
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflow_json_path: str = "workflows/flux_klein.json",
        input_node_descriptions: Optional[List[Dict[str, Any]]] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        初始化 Flux 图片生成器
        
        Args:
            base_url: ComfyUI 服务地址
            workflow_json_path: Flux 工作流 JSON 文件路径
            input_node_descriptions: 输入节点描述列表
            rate_limiter: 速率限制器
        """
        # 默认的 Flux 工作流输入节点配置（使用整数节点 ID）
        default_input_descriptions = input_node_descriptions or [
            {"node_id": 103, "field": "text_g", "source": "prompt"},
            {"node_id": 104, "field": "text_l", "source": "prompt"},
            {"node_id": 105, "field": "image", "source": "reference_image", "max_count": 4},
        ]
        
        self.runner = ComfyUIWorkflowRunner(
            base_url=base_url,
            workflow_json_path=workflow_json_path,
            input_node_descriptions=default_input_descriptions,
            output_node_ids=[],  # 自动检测
            rate_limiter=rate_limiter,
        )
        
        logger.info(f"Initialized ImageGeneratorComfyUIFlux with workflow: {workflow_json_path}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], rate_limiter: Optional[RateLimiter] = None) -> "ImageGeneratorComfyUIFlux":
        """从配置字典创建实例"""
        return cls(
            base_url=config.get("base_url", "http://127.0.0.1:8188"),
            workflow_json_path=config.get("workflow_json_path", "workflows/flux_klein.json"),
            input_node_descriptions=config.get("input_node_descriptions"),
            rate_limiter=rate_limiter,
        )
    
    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = None,
        **kwargs,
    ) -> ImageOutput:
        """
        生成单张图片
        
        Args:
            prompt: 图片描述文本
            reference_image_paths: 参考图片路径列表（用于风格一致性）
            **kwargs: 其他参数（seed, steps 等，传递给工作流）
        
        Returns:
            ImageOutput: 生成的图片输出
        """
        logger.info("Generating image with Flux.klein...")
        
        # 执行工作流
        outputs = await self.runner.run(
            prompt=prompt,
            reference_image_paths=reference_image_paths or [],
            **kwargs,
        )
        
        # 提取输出路径
        output_paths = self.runner.get_output_paths(outputs)
        
        if not output_paths:
            raise RuntimeError("No image output generated")
        
        # 获取第一个输出节点的结果
        first_output = list(output_paths.values())[0]
        
        if isinstance(first_output, list) and len(first_output) > 0:
            # 输出是图片路径列表
            image_path = first_output[0]
            return ImageOutput(fmt="pil", ext="png", data=image_path)
        elif isinstance(first_output, str):
            # 输出是单个文件路径
            return ImageOutput(fmt="pil", ext="png", data=first_output)
        else:
            raise RuntimeError(f"Unexpected output format: {first_output}")
    
    async def generate_images(
        self,
        prompts: List[str],
        reference_image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> List[ImageOutput]:
        """
        批量生成图片（并行）
        
        Args:
            prompts: 图片描述文本列表
            reference_image_paths: 参考图片路径列表
            **kwargs: 其他参数
        
        Returns:
            图片输出列表
        """
        import asyncio
        
        tasks = [
            self.generate_single_image(prompt=prompt, reference_image_paths=reference_image_paths, **kwargs)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤错误
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate image {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results


# 为了保持向后兼容，也导出通用类
ComfyUIImageGenerator = ImageGeneratorComfyUIFlux
