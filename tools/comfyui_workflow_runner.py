"""
ComfyUI Workflow Runner: 通用的 ComfyUI 工作流执行器

支持通过配置文件定义输入节点，用户只需提供：
1. 工作流 JSON

Usage:
    runner = ComfyUIWorkflowRunner.from_config(
        base_url="http://127.0.0.1:8188",
    )
    
    result = await runner.run(workflow={})
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import aiohttp
from websockets.sync.client import connect as ws_connect
from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

@dataclass
class ComfyUIWorkflowConfig:
    """ComfyUI 工作流配置"""
    base_url: str
    output_node_ids: list[int | str] = field(default_factory=list)  # 输出节点 ID 列表
    client_id: str | None = None


class ComfyUIWorkflowRunner:
    """
    ComfyUI 工作流通用执行器
    
    核心能力：
    1. run 提交到队列
    2. 通过 WebSocket 获取执行进度和结果
    3. 上传图片
    """
    
    # 常见的输出节点类型
    IMAGE_OUTPUT_TYPES = {"SaveImage", "PreviewImage"}
    VIDEO_OUTPUT_TYPES = {"VHS_VideoCombine", "VideoCombine", "SaveVideo"}
    
    def __init__(
        self,
        base_url: str,
        rate_limiter: RateLimiter | None = None,
        client_id: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.client_id = client_id or str(uuid.uuid4())
        self.output_node_ids = []
        
    def load_workflow(self, workflow_path: str) -> dict[str, Any]:
        """加载工作流 JSON 文件"""

        path = Path(workflow_path)
        if not path.exists():
            # 尝试相对于当前文件目录
            path = Path(__file__).parent.parent / workflow_path
        
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        
        logger.info(f"Loaded workflow from {path}")
        return workflow
        
    async def upload_image(
        self,
        image_path: str,
    ) -> str:
        """上传图片"""
        
        url = f"{self.base_url}/upload/image"
        
        # 根据文件扩展名识别 MIME 类型
        ext = Path(image_path).suffix.lower()
        content_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")
        filename = Path(image_path).name
        
        async with aiohttp.ClientSession() as session:
            with open(image_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("image", f, filename=filename, content_type=content_type)
                data.add_field("type", "input")
                data.add_field("overwrite", "true")

                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to upload image: {error_text}")
                    
                    result = await response.json()
                    return result["name"]

    async def run(
        self,
        workflow: dict,
        output_node_ids: List[int | str],
    ) -> dict[int | str, Any]:
        """
        执行工作流
        
        Args:
            workflow: 工作流json
            output_node_ids: 输出节点 ID 列表
        
        Returns:
            执行结果字典，包含输出文件路径
        """
        
        # 应用速率限制
        if self.rate_limiter:
            await self.rate_limiter.acquire()
            
        self.output_node_ids = output_node_ids
        
        # 执行工作流
        result = await self._execute_workflow(workflow)
        
        return result
    
    async def _execute_workflow(self, workflow: dict[str, Any]) -> dict[int | str, Any]:
        """执行工作流"""

        prompt_id = await self._queue_prompt(workflow)
        logger.info(f"Comfyui Queued prompt: {prompt_id}")
        
        # 等待执行完成
        outputs = await self._wait_for_outputs(prompt_id)
        
        return outputs
    
    async def _queue_prompt(self, workflow: dict[str, Any]) -> str:
        """将工作流加入队列"""

        url = f"{self.base_url}/prompt"
        
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to queue prompt: {error_text}")
                
                result = await response.json()
                return result["prompt_id"]
    
    async def _wait_for_outputs(self, prompt_id: str) -> dict[int | str, Any]:
        """
        等待并获取输出
        
        注意：新格式中节点 ID 是整数，outputs 字典的 key 也应该是整数
        """

        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?clientId={self.client_id}"
        
        outputs: dict[int | str, Any] = {}
        
        # 使用线程方式连接 WebSocket（asyncio 版本需要在单独线程中运行）
        import threading
        result_holder: dict[str, Any] = {"outputs": None, "error": None, "finished": False}
        
        # 标准化输出节点 ID（统一为整数）
        normalized_output_ids: set[int | str] = set()
        for oid in self.output_node_ids:
            normalized_output_ids.add(oid)
        
        def ws_receiver():
            try:
                with ws_connect(ws_url) as ws:
                    while not result_holder["finished"]:
                        message = ws.recv()
                        
                        # 处理二进制消息（跳过图片等二进制数据）
                        if isinstance(message, bytes):
                            continue
                        
                        data = json.loads(message)
                        msg_type = data.get("type")
                        msg_data = data.get("data", {})
                        
                        if msg_data.get("prompt_id") != prompt_id:
                            continue
                        
                        # 处理执行进度
                        if msg_type == "executing":
                            node = msg_data.get("node")
                            if node:
                                logger.info(f"正在执行节点：{node}")
                                
                        elif msg_type == "progress":
                            value = msg_data.get("value", 0)
                            max_val = msg_data.get("max", 100)
                            logger.info(f"进度：{value}/{max_val}")
                        
                        # 处理执行完成
                        elif msg_type == "executed":
                            node_id = msg_data.get("node")
                            output = msg_data.get("output", {})
                            if output:
                                outputs[node_id] = output
                        
                        elif msg_type == "execution_success":
                            logger.info("任务成功完成！")
                            result_holder["finished"] = True
                            break
        
                        # 处理错误
                        elif msg_type == "execution_error":
                            result_holder["error"] = msg_data
                            result_holder["finished"] = True
                            break

                        # 检查是否全部完成
                        if all(oid in outputs for oid in normalized_output_ids):
                            result_holder["outputs"] = outputs
                            result_holder["finished"] = True
                            break
                            
            except Exception as e:
                result_holder["error"] = str(e)
                result_holder["finished"] = True
        
        # 在后台线程运行 WebSocket 接收
        ws_thread = threading.Thread(target=ws_receiver)
        ws_thread.start()
        
        # 等待完成（带超时）
        timeout = 300  # 5 分钟超时
        start_time = asyncio.get_event_loop().time()
        while not result_holder["finished"]:
            await asyncio.sleep(0.5)
            if asyncio.get_event_loop().time() - start_time > timeout:
                result_holder["finished"] = True
                result_holder["error"] = "Timeout waiting for workflow completion"
        
        ws_thread.join(timeout=5)
        
        if result_holder["error"]:
            raise RuntimeError(f"Workflow execution error: {result_holder['error']}")
        
        return result_holder["outputs"] or outputs
    
    def get_output_paths(self, outputs: dict[int | str, Any]) -> dict[int | str, Any]:
        """从执行结果中提取输出文件路径"""
        paths: dict[int | str, Any] = {}
        
        t = f"{self.base_url}/view?filename={{filename}}&subfolder={{subfolder}}&type={{type}}"
        
        for node_id, output in outputs.items():
            if not output:
                continue

            # 检查输出内容
            if "images" in output:
                paths[node_id] = t.format(**output["images"][0])
            elif "gifs" in output:
                paths[node_id] = output["gifs"]
            elif "videos" in output:
                paths[node_id] = output["videos"]
            elif "ui" in output:
                # ComfyUI 的 UI 输出格式
                ui_data = output["ui"]
                if isinstance(ui_data, dict):
                    for key, value in ui_data.items():
                        if isinstance(value, list):
                            paths[f"{node_id}:{key}"] = value
                elif isinstance(ui_data, list):
                    paths[node_id] = ui_data
        
        return paths
