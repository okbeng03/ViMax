"""
ImageGeneratorComfyUIFlux: 使用 Flux.klein 模型生成图片

通过 ComfyUI 工作流调用本地部署的 Flux.klein 模型

Usage:
    # 方式1: 直接实例化
    generator = ImageGeneratorComfyUIFlux(
        base_url="http://127.0.0.1:8188"
    )
    
    image = await generator.generate_single_image(
        prompt="A cute cat",
        reference_image_paths=["/path/to/style.png"]
    )
    
    # 方式2: 从配置实例化
    generator = ImageGeneratorComfyUIFlux.from_config(config)
"""

import logging
import json
import uuid
import random
from typing import Any, List, Optional
from tenacity import retry, stop_after_attempt
from interfaces.image_output import ImageOutput
from tools.comfyui_workflow_runner import ComfyUIWorkflowRunner
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)

text_to_image_workflow_path = "workflows/flux2_klein_text_to_image.json"
text_to_image_ui_workflow_path = "workflows/flux2_klein_text_to_image_ui.json"
image_to_image_workflow_path = "workflows/flux2_klein_8image.json"
image_to_image_ui_workflow_path = "workflows/flux2_klein_8image_ui.json"
qwen_image_edit_workflow_path = "workflows/qwen_image_edit.json"
qwen_image_edit_ui_workflow_path = "workflows/qwen_image_edit_ui.json"
qwen_image_edit_camera_workflow_path = "workflows/qwen-edit-multiple-angles.json"
qwen_image_edit_camera_ui_workflow_path = "workflows/qwen-edit-multiple-angles_ui.json"
text_to_image_output_node_ids = ["9"]
image_to_image_output_node_ids = ["94"]
qwen_image_edit_output_node_ids = ["14"]
qwen_image_edit_camera_output_node_ids = ["117"]
max_noise = 2**50 - 1

class ImageGeneratorComfyUIFlux:
    """
    Flux.klein 图片生成器
    
    基于 ComfyUI 工作流，支持：
    - 文生图
    - 图生图：多参考图片输入（支持多张）
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
    
    async def load_qwen_edit_workflow(self, runner: ComfyUIWorkflowRunner, prompt: str, reference_image_paths: List[str] = None, width: int = 2048, height: int = 2048) -> dict[str, Any]:
        """加载Qwen编辑工作流"""
        
        workflow = runner.load_workflow(qwen_image_edit_workflow_path)
        workflow["14"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["8"]["inputs"]["prompt"] = prompt
        workflow["22"]["inputs"]["scale_to_length"] = width
        workflow["7"]["inputs"]["seed"] = random.randint(1, max_noise)
        
        image1 = reference_image_paths[0]
        image1_name = await runner.upload_image(image1)
        workflow["10"]["inputs"]["image"] = image1_name
        
        image2 = reference_image_paths[1]
        image2_name = await runner.upload_image(image2)
        workflow["11"]["inputs"]["image"] = image2_name
        
        # ui
        ui_workflow = runner.load_workflow(qwen_image_edit_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 14), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())
        positive_node = next((node for node in nodes if node["id"] == 8), None)
        positive_node["widgets_values"][0] = prompt
        scale_node = next((node for node in nodes if node["id"] == 22), None)
        scale_node["widgets_values"][7] = width
        noise_node = next((node for node in nodes if node["id"] == 7), None)
        noise_node["widgets_values"][0] = random.randint(1, max_noise)
        
        return workflow, ui_workflow

    async def load_qwen_edit_camera_workflow(self, runner: ComfyUIWorkflowRunner, prompt: str, reference_image_paths: List[str] = None, width: int = 2048, height: int = 2048) -> dict[str, Any]:
        """加载Qwen场景一致性编辑工作流"""
        
        workflow = runner.load_workflow(qwen_image_edit_camera_workflow_path)
        workflow["117"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["112"]["inputs"]["prompt"] = prompt
        workflow["116"]["inputs"]["longer_edge"] = width
        workflow["106"]["inputs"]["seed"] = random.randint(1, max_noise)
        
        image1 = reference_image_paths[0]
        image1_name = await runner.upload_image(image1)
        workflow["41"]["inputs"]["image"] = image1_name
        
        # image2 = reference_image_paths[1]
        # image2_name = await runner.upload_image(image2)
        # workflow["11"]["inputs"]["image"] = image2_name
        
        # ui
        ui_workflow = runner.load_workflow(qwen_image_edit_camera_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 117), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())
        positive_node = next((node for node in nodes if node["id"] == 112), None)
        positive_node["widgets_values"][0] = prompt
        scale_node = next((node for node in nodes if node["id"] == 116), None)
        scale_node["widgets_values"][0] = width
        noise_node = next((node for node in nodes if node["id"] == 106), None)
        noise_node["widgets_values"][0] = random.randint(1, max_noise)
        image_node = next((node for node in nodes if node["id"] == 41), None)
        image_node["widgets_values"][0] = image1_name
        
        return workflow, ui_workflow
    
    
    async def load_t2i_workflow(self, runner: ComfyUIWorkflowRunner, prompt: str, width: int, height: int) -> dict[str, Any]:
        """加载文生图工作流"""
    
        workflow = runner.load_workflow(text_to_image_workflow_path)
        workflow["9"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["76"]["inputs"]["value"] = prompt
        workflow["75:68"]["inputs"]["value"] = width
        workflow["75:69"]["inputs"]["value"] = height
        workflow["75:73"]["inputs"]["noise_seed"] = random.randint(1, max_noise)
        
        # ui
        ui_workflow = runner.load_workflow(text_to_image_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 9), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())
        positive_node = next((node for node in nodes if node["id"] == 76), None)
        positive_node["widgets_values"][0] = prompt
        
        subgraph_nodes = ui_workflow["definitions"]["subgraphs"][0]["nodes"]
        width_node = next((node for node in subgraph_nodes if node["id"] == 68), None)
        width_node["widgets_values"][0] = width
        height_node = next((node for node in subgraph_nodes if node["id"] == 69), None)
        height_node["widgets_values"][0] = height
        
        return workflow, ui_workflow
    
    async def load_i2i_workflow(self, runner: ComfyUIWorkflowRunner, prompt: str, reference_image_paths: List[str] = None, width: int = 2048, height: int = 2048) -> dict[str, Any]:
        """加载图生图工作流"""

        workflow = runner.load_workflow(image_to_image_workflow_path)
        workflow["94"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow["92:113"]["inputs"]["text"] = prompt
        workflow["92:197"]["inputs"]["value"] = height
        workflow["92:199"]["inputs"]["value"] = width
        workflow["92:105"]["inputs"]["noise_seed"] = random.randint(1, max_noise)
        
        
        # 删除对应节点
        reference_images_len = len(reference_image_paths)
        image_nodes = ["76", "81", "158", "157", "160", "159", "162", "161"]
        remove_image_nodes = image_nodes[reference_images_len:]
        load_image_nodes = image_nodes[:reference_images_len]
        group_image_nodes = [
            {
                "image": "92:110",
                "vae_encode": "92:165",
                "positive_latent": "92:166",
                "negative_latent": "92:167",
            },
            {
                "image": "92:168",
                "vae_encode": "92:169",
                "positive_latent": "92:170",
                "negative_latent": "92:171",
            },
            {
                "image": "92:172",
                "vae_encode": "92:173",
                "positive_latent": "92:174",
                "negative_latent": "92:175",
            },
            {
                "image": "92:176",
                "vae_encode": "92:177",
                "positive_latent": "92:178",
                "negative_latent": "92:179",
            },
            {
                "image": "92:180",
                "vae_encode": "92:181",
                "positive_latent": "92:182",
                "negative_latent": "92:183",
            },
            {
                "image": "92:184",
                "vae_encode": "92:185",
                "positive_latent": "92:186",
                "negative_latent": "92:187",
            },
            {
                "image": "92:188",
                "vae_encode": "92:189",
                "positive_latent": "92:190",
                "negative_latent": "92:191",
            },
            {
                "image": "92:192",
                "vae_encode": "92:193",
                "positive_latent": "92:194",
                "negative_latent": "92:195",
            },
        ]
        retain_group_image_nodes = group_image_nodes[:reference_images_len]
        remove_group_image_nodes = group_image_nodes[reference_images_len:]
        
        for i, image_path in enumerate(reference_image_paths):
            image_name = await runner.upload_image(image_path)
            workflow[load_image_nodes[i]]["inputs"]["image"] = image_name

        for node_id in remove_image_nodes:
            workflow.pop(node_id, None)
        for group in remove_group_image_nodes:
            for node_id in group.values():
                workflow.pop(node_id, None)
            
        workflow["92:114"]["inputs"]["positive"] = [retain_group_image_nodes[-1]["positive_latent"], 0]
        workflow["92:114"]["inputs"]["negative"] = [retain_group_image_nodes[-1]["negative_latent"], 0]
        
        #  ui
        ui_workflow = runner.load_workflow(image_to_image_ui_workflow_path)
        nodes = ui_workflow["nodes"]
        output_node = next((node for node in nodes if node["id"] == 94), None)
        output_node["widgets_values"][0] = str(uuid.uuid4())
        
        for i, _ in enumerate(reference_image_paths):
            image_node = next((node for node in nodes if node["id"] == int(load_image_nodes[i])), None)
            image_node["widgets_values"][0] = workflow[load_image_nodes[i]]["inputs"]["image"]
            
        for node_id in remove_image_nodes:
            image_node = next((node for node in nodes if node["id"] == int(node_id)), None)
            image_node["mode"] = 4
        
        subgraph_nodes = ui_workflow["definitions"]["subgraphs"][0]["nodes"]
        width_node = next((node for node in subgraph_nodes if node["id"] == 199), None)
        width_node["widgets_values"][0] = width
        height_node = next((node for node in subgraph_nodes if node["id"] == 197), None)
        height_node["widgets_values"][0] = height
        prompt_node = next((node for node in subgraph_nodes if node["id"] == 113), None)
        prompt_node["widgets_values"][0] = prompt        
        for group in remove_group_image_nodes:
            for group_id in group.values():
                node_id = group_id.split(":")[1]
                remove_node = next((node for node in subgraph_nodes if node["id"] == int(node_id)), None)
                remove_node["mode"] = 4

        return workflow, ui_workflow
    
    async def generate_single_image(
        self,
        prompt: str,
        *,
        reference_image_paths: List[str] = None,
        size: Optional[str] = "2048x2048",
        workflow_name: Optional[str] = None,
        enable_schedule_mode: Optional[bool] = False,
        **kwargs,
    ) -> ImageOutput | None:
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
        
        runner = ComfyUIWorkflowRunner(
            base_url=self.base_url,
            rate_limiter=self.rate_limiter,
        )
        workflow = {}
        
        # 从size解析width、height
        if size:
            width, height = map(int, size.split("x"))
        
        if workflow_name == "qwen_edit":
            logger.info("==========Using Qwen edit workflow================")
            workflow, ui_workflow = await self.load_qwen_edit_workflow(runner=runner, prompt=prompt, reference_image_paths=reference_image_paths, width=width, height=height)
            workflow_path = qwen_image_edit_workflow_path
            output_node_ids = qwen_image_edit_output_node_ids
        elif workflow_name == "qwen_edit_camera":
            logger.info("==========Using Qwen edit camera workflow================")
            workflow, ui_workflow = await self.load_qwen_edit_camera_workflow(runner=runner, prompt=prompt, reference_image_paths=reference_image_paths, width=width, height=height)
            workflow_path = qwen_image_edit_camera_workflow_path
            output_node_ids = qwen_image_edit_camera_output_node_ids
        else:
            if reference_image_paths:
                logger.info("==========Using reference images for style consistency================")
                workflow, ui_workflow = await self.load_i2i_workflow(runner=runner, prompt=prompt, reference_image_paths=reference_image_paths, width=width, height=height)
                workflow_path = image_to_image_workflow_path
                output_node_ids = image_to_image_output_node_ids
            else:
                workflow, ui_workflow = await self.load_t2i_workflow(runner=runner, prompt=prompt, width=width, height=height)
                workflow_path = text_to_image_workflow_path
                output_node_ids = text_to_image_output_node_ids
        # print("========================\n", json.dumps(workflow, indent=4), "\n========================")

        ready = await runner.prepare(
            workflow_path=workflow_path,
            workflow=workflow,
            output_node_ids=output_node_ids,
            ui_workflow=ui_workflow,
            enable_schedule_mode=enable_schedule_mode,
        )

        if not ready:
            return

        # 执行工作流
        outputs = await runner.run(
            workflow_path=workflow_path,
            workflow=workflow,
            output_node_ids=output_node_ids,
            ui_workflow=ui_workflow,
        )
        # 提取输出路径
        output_paths = runner.get_output_paths(outputs)
        
        if not output_paths:
            raise RuntimeError("No image output generated")
        
        # 获取第一个输出节点的结果
        first_output = list(output_paths.values())[0]

        if isinstance(first_output, list) and len(first_output) > 0:
            # 输出是图片路径列表
            image_path = first_output[0]
            return ImageOutput(fmt="url", ext="png", data=image_path)
        elif isinstance(first_output, str):
            # 输出是单个文件路径
            return ImageOutput(fmt="url", ext="png", data=first_output)
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
