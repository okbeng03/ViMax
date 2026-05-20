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
import time
import uuid
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import aiohttp
from websockets.sync.client import connect as ws_connect
from utils.rate_limiter import RateLimiter
from configs.config import working_dir

logger = logging.getLogger(__name__)


@dataclass
class ComfyUIWorkflowConfig:
    """ComfyUI 工作流配置"""
    base_url: str
    output_node_ids: list[int | str] = field(default_factory=list)  # 输出节点 ID 列表
    client_id: str | None = None


@dataclass
class WorkflowTask:
    """工作流任务"""
    workflow_path: str
    workflow: dict[str, Any]
    output_node_ids: list[int | str]
    timeout: int
    ui_workflow: dict[str, Any] | None = None
    future: asyncio.Future[Any] | None = None
    
    def __post_init__(self):
        if self.future is None:
            loop = asyncio.get_running_loop()
            self.future = loop.create_future()


# ============ 全局队列（所有实例共享） ============
_global_task_queue: asyncio.Queue[WorkflowTask] | None = None
_global_worker_task: asyncio.Task[None] | None = None
_global_is_running = False
_global_queue_lock: asyncio.Lock | None = None


def _get_global_queue() -> asyncio.Queue[WorkflowTask]:
    """获取或创建全局队列"""
    global _global_task_queue
    if _global_task_queue is None:
        _global_task_queue = asyncio.Queue()
    return _global_task_queue


def _get_global_lock() -> asyncio.Lock:
    """获取或创建全局锁"""
    global _global_queue_lock
    if _global_queue_lock is None:
        _global_queue_lock = asyncio.Lock()
    return _global_queue_lock


class ComfyUIWorkflowRunner:
    """
    ComfyUI 工作流通用执行器
    
    核心能力：
    1. run 提交到全局队列，顺序执行（避免 GPU 并发冲突）
    2. 通过 WebSocket 获取执行进度和结果
    3. 上传图片
    
    注意：队列是全局的，所有实例共享同一个队列
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
        # 生成随机文件名
        filename = f"{uuid.uuid4()}{ext}"
        
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
                
    async def upload_audio(
        self,
        audio_path: str,
    ) -> str:
        """上传音频"""
        
        url = f"{self.base_url}/upload/image"
        
        # 根据文件扩展名识别 MIME 类型
        ext = Path(audio_path).suffix.lower()
        content_type_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }

        content_type = content_type_map.get(ext, "application/octet-stream")
        # 生成随机文件名
        filename = f"{uuid.uuid4()}{ext}"
        
        async with aiohttp.ClientSession() as session:
            with open(audio_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("image", f, filename=filename, content_type=content_type)
                data.add_field("type", "input")
                data.add_field("overwrite", "true")

                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to upload audio: {error_text}")
                    
                    result = await response.json()
                    return result["name"]

    async def run(
        self,
        workflow_path: str,
        workflow: dict[str, Any],
        output_node_ids: List[int | str],
        *,
        timeout: int = 1000,
        ui_workflow: dict[str, Any] | None = None,
    ) -> dict[int | str, Any]:
        """
        执行工作流（加入全局队列，顺序执行）
        
        Args:
            workflow: 工作流json
            output_node_ids: 输出节点 ID 列表
            timeout: 超时时间
        
        Returns:
            执行结果字典，包含输出文件路径
        """
        
        global _global_worker_task, _global_is_running
        
        # 启动队列处理器（如果尚未启动）
        async with _get_global_lock():
            if _global_worker_task is None or _global_worker_task.done():
                _global_is_running = True
                _global_worker_task = asyncio.create_task(self._global_queue_worker())
        
        # 创建任务并加入全局队列
        task = WorkflowTask(
            workflow_path=workflow_path,
            workflow=workflow,
            ui_workflow=ui_workflow,
            output_node_ids=output_node_ids,
            timeout=timeout,
        )
        
        queue = _get_global_queue()
        await queue.put(task)
        logger.info(f"工作流已加入全局队列，当前队列长度: {queue.qsize()}")
        
        # 等待任务执行完成（future 已在 __post_init__ 中初始化）
        assert task.future is not None, "Future should be initialized in __post_init__"
        result = await task.future
        
        return result
    
    async def _global_queue_worker(self):
        """全局队列工作器：顺序处理队列中的任务"""
        global _global_is_running
        
        logger.info("全局队列工作器已启动")
        
        while _global_is_running:
            try:
                # 等待下一个任务
                queue = _get_global_queue()
                task = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                logger.info(f"开始执行工作流{task.workflow_path}...")
                
                # 应用速率限制
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                
                # 更新输出节点 ID
                self.output_node_ids = task.output_node_ids
                
                # 执行工作流
                try:
                    result = await self._execute_workflow(task)
                    assert task.future is not None
                    task.future.set_result(result)
                except Exception as e:
                    assert task.future is not None
                    task.future.set_exception(e)
                    logger.error(f"工作流执行失败: {e}")
                
                # 标记任务完成
                queue.task_done()
                logger.info("工作流执行完成")
                
                # 等待30秒，保证ComfyUI环境释放
                logger.info("等待30秒后开始下一个任务...")
                await asyncio.sleep(30)
                
            except asyncio.TimeoutError:
                # 没有新任务，继续循环检查是否应该退出
                continue
            except asyncio.CancelledError:
                # 获取更多取消上下文信息
                import traceback
                cancel_trace = traceback.format_stack()
                logger.warning(f"全局队列工作器被取消。堆栈信息: {''.join(cancel_trace[-5:])}")
                _global_is_running = False
                break
            except Exception as e:
                logger.error(f"队列处理异常: {e}")
        
        logger.info("全局队列工作器已停止")
    
    @staticmethod
    def stop_worker():
        """停止全局队列工作器"""
        global _global_is_running, _global_worker_task
        _global_is_running = False
        if _global_worker_task and not _global_worker_task.done():
            _global_worker_task.cancel()
    
    @staticmethod
    async def wait_queue_empty():
        """等待全局队列中的所有任务执行完成"""
        queue = _get_global_queue()
        await queue.join()
    
    @staticmethod
    def get_queue_size() -> int:
        """获取当前全局队列长度"""
        queue = _get_global_queue()
        return queue.qsize()
    
    async def _execute_workflow(self, task: WorkflowTask) -> dict[int | str, Any]:
        """执行工作流"""

        # 记录工作流开始执行时间
        workflow_start_time = time.time()
        prompt_id = await self._queue_prompt(task.workflow)
        logger.info(f"Comfyui Queued prompt: {prompt_id}. 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow_start_time))}")
        workflow_name = os.path.splitext(os.path.basename(task.workflow_path))[0] if task.workflow_path else ""
        workflow_path = os.path.join(f"{working_dir}/workflows", f"{workflow_name}_{prompt_id}.json")
        
        with open(workflow_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(task.workflow, ensure_ascii=False, indent=4))
            
        if task.ui_workflow:
            with open(os.path.join(f"{working_dir}/workflows", f"{prompt_id}_ui.json"), 'w', encoding='utf-8') as f:
                f.write(json.dumps(task.ui_workflow, ensure_ascii=False, indent=4))
        
        # 等待执行完成，传入开始时间用于计算耗时
        outputs = await self._wait_for_outputs(prompt_id, task.timeout, workflow_start_time)
        
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
    
    async def _wait_for_outputs(self, prompt_id: str, timeout: int, workflow_start_time) -> dict[int | str, Any]:
        """
        等待并获取输出
        
        注意：新格式中节点 ID 是整数，outputs 字典的 key 也应该是整数
        
        Args:
            prompt_id: prompt 标识符
            timeout: 超时时间
            workflow_start_time: 工作流开始执行的时间戳（用于计算总耗时）
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
            
        ws_connect_time = None
        
        def ws_receiver():
            nonlocal ws_connect_time
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
                            
                            if not ws_connect_time:
                                ws_connect_time = time.time()
                        
                        # 处理执行完成
                        elif msg_type == "executed":
                            node_id = msg_data.get("node")
                            output = msg_data.get("output", {})
                            if output:
                                outputs[node_id] = output
                        
                        elif msg_type == "execution_success":
                            # 计算并输出耗时
                            end_time = time.time()
                            if workflow_start_time:
                                total_duration = end_time - workflow_start_time
                                logger.info(f"Prompt {prompt_id} 执行成功，总耗时: {total_duration:.2f} 秒")
                            else:
                                logger.info(f"Prompt {prompt_id} 执行成功")
                            result_holder["finished"] = True
                            break
        
                        # 处理错误
                        elif msg_type == "execution_error":
                            result_holder["error"] = msg_data
                            result_holder["finished"] = True
                            break

                        # 检查是否全部完成
                        if all(oid in outputs for oid in normalized_output_ids):
                            result_holder["outputs"] = {oid: outputs[oid] for oid in normalized_output_ids}
                            result_holder["finished"] = True
                            logger.info(f"Prompt {prompt_id} 执行完成。结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
                            break
                            
            except Exception as e:
                result_holder["error"] = str(e)
                result_holder["finished"] = True
        
        # 在后台线程运行 WebSocket 接收
        ws_thread = threading.Thread(target=ws_receiver)
        ws_thread.start()
        
        # 等待完成（带超时）
        timeout = timeout  # 5 分钟超时

        while not result_holder["finished"]:
            await asyncio.sleep(0.5)
            if ws_connect_time and time.time() - ws_connect_time > timeout:
                result_holder["finished"] = True
                result_holder["error"] = f"Timeout waiting for workflow completion:: {timeout}s。结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"
        
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
                paths[node_id] = t.format(**output["gifs"][0])
            elif "videos" in output:
                paths[node_id] = t.format(**output["videos"][0])
            elif "audio" in output:
                paths[node_id] = t.format(**output["audio"][0])
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
