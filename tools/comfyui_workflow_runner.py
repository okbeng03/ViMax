"""
ComfyUI Workflow Runner: 通用的 ComfyUI 工作流执行器

支持通过配置文件定义输入节点，用户只需提供：
1. 工作流 JSON 文件路径
2. 输入节点描述（哪些节点是输入，如何填充值）

Usage:
    runner = ComfyUIWorkflowRunner.from_config(
        base_url="http://127.0.0.1:8188",
        workflow_json_path="workflows/my_workflow.json",
        input_node_descriptions=[
            {"node_id": "3", "field": "text", "source": "prompt"},
            {"node_id": "5", "field": "image", "source": "reference_image", "max_count": 4},
        ]
    )
    
    result = await runner.run(
        prompt="hello world",
        reference_image_paths=["/path/to/img1.png", "/path/to/img2.png"]
    )
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
from PIL import Image
from websockets.sync.client import connect as ws_connect

from interfaces.image_output import ImageOutput
from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


class NodeInputSource(str, Enum):
    """输入节点的数据源类型"""
    PROMPT = "prompt"                    # 文本 prompt
    REFERENCE_IMAGE = "reference_image"  # 参考图片（多张）
    FRAME = "frame"                     # 帧图片（按索引）
    SEED = "seed"                       # 随机种子
    STEPS = "steps"                     # 采样步数
    CFG = "cfg"                         # CFG 值
    WIDTH = "width"                     # 宽度
    HEIGHT = "height"                  # 高度


@dataclass
class ComfyUIWorkflowMetadata:
    """ComfyUI 工作流元数据（新格式）"""
    id: str
    revision: int = 0
    last_node_id: int = 0
    last_link_id: int = 0


@dataclass
class NodeInputDescription:
    """单个输入节点的描述
    
    注意: node_id 现在支持整数或字符串类型（兼容旧格式）
    """
    node_id: int | str                         # ComfyUI 节点 ID（整数或字符串）
    field: str                                   # 输入字段名
    source: NodeInputSource                      # 数据源类型
    max_count: int | None = None                 # 对于多输入，最大数量限制
    index: int | None = None                     # 对于 FRAME 类型，指定帧索引
    default: Any | None = None                   # 默认值


@dataclass
class ComfyUIWorkflowConfig:
    """ComfyUI 工作流配置"""
    base_url: str
    workflow_json_path: str
    input_node_descriptions: list[NodeInputDescription]
    output_node_ids: list[int | str] = field(default_factory=list)  # 输出节点 ID 列表
    client_id: str | None = None


class ComfyUIWorkflowRunner:
    """
    ComfyUI 工作流通用执行器
    
    支持两种工作流 JSON 格式：
    1. 旧格式: 节点直接作为字典 key (如 {"3": {...}, "5": {...}})
    2. 新格式: 节点在 nodes 数组中 (如 {"id": "...", "nodes": [{"id": 103, ...}]})
    
    核心能力：
    1. 从 JSON 文件加载工作流（自动检测格式）
    2. 根据 input_node_descriptions 动态填充输入节点
    3. 通过 WebSocket 获取执行进度和结果
    4. 支持图片、视频等多种输出类型
    """
    
    # 常见的输出节点类型
    IMAGE_OUTPUT_TYPES = {"SaveImage", "PreviewImage", "LoadImage"}
    VIDEO_OUTPUT_TYPES = {"VHS_VideoCombine", "VideoCombine"}
    
    def __init__(
        self,
        base_url: str,
        workflow_json_path: str,
        input_node_descriptions: list[dict[str, Any] | NodeInputDescription],
        output_node_ids: list[int | str] | None = None,
        client_id: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.workflow_json_path = workflow_json_path
        self.rate_limiter = rate_limiter
        self.client_id = client_id or str(uuid.uuid4())
        
        # 解析输入节点描述
        self.input_node_descriptions: list[NodeInputDescription] = []
        for desc in input_node_descriptions:
            if isinstance(desc, dict):
                desc_copy = dict(desc)  # 创建副本避免修改原字典
                desc_copy["source"] = NodeInputSource(desc_copy["source"])
                self.input_node_descriptions.append(NodeInputDescription(**desc_copy))
            else:
                self.input_node_descriptions.append(desc)
        
        # 解析输出节点（如果未指定，尝试自动检测）
        self.output_node_ids = output_node_ids or []
        
        # 加载工作流
        self.workflow = self._load_workflow()
        
        # 解析工作流格式
        self._parse_workflow_format()
        
        # 验证工作流结构
        self._validate_workflow()
        
    @classmethod
    def from_config(cls, config: Dict[str, Any], rate_limiter: Optional[RateLimiter] = None) -> "ComfyUIWorkflowRunner":
        """从配置字典创建实例"""
        # 解析输入节点描述
        input_node_descriptions = []
        for desc in config.get("input_node_descriptions", []):
            if isinstance(desc.get("source"), str):
                desc["source"] = NodeInputSource(desc["source"])
            input_node_descriptions.append(desc)
        
        return cls(
            base_url=config.get("base_url", "http://127.0.0.1:8188"),
            workflow_json_path=config["workflow_json_path"],
            input_node_descriptions=input_node_descriptions,
            output_node_ids=config.get("output_node_ids", []),
            client_id=config.get("client_id"),
            rate_limiter=rate_limiter,
        )
    
    def _load_workflow(self) -> Dict[str, Any]:
        """加载工作流 JSON 文件"""
        path = Path(self.workflow_json_path)
        if not path.exists():
            # 尝试相对于当前文件目录
            path = Path(__file__).parent.parent / self.workflow_json_path
        
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {self.workflow_json_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        
        logger.info(f"Loaded workflow from {path}")
        return workflow
    
    def _parse_workflow_format(self):
        """
        解析工作流格式，检测是旧格式还是新格式
        
        新格式 (nodes 数组):
            {
                "id": "...",
                "nodes": [{"id": 103, ...}, ...]
            }
        
        旧格式 (直接节点):
            {"103": {...}, "105": {...}, ...}
        """
        # 检查是否是新的 nodes 数组格式
        if "nodes" in self.workflow and isinstance(self.workflow.get("nodes"), list):
            self.is_new_format = True
            self.metadata = ComfyUIWorkflowMetadata(
                id=self.workflow.get("id", ""),
                revision=self.workflow.get("revision", 0),
                last_node_id=self.workflow.get("last_node_id", 0),
                last_link_id=self.workflow.get("last_link_id", 0),
            )
            # 构建节点 ID 到节点的映射
            self._nodes_map: dict[int, dict[str, Any]] = {}
            for node in self.workflow.get("nodes", []):
                node_id = node.get("id")
                if node_id is not None:
                    self._nodes_map[int(node_id)] = node
            
            logger.info(f"Detected new format workflow with {len(self._nodes_map)} nodes")
        else:
            self.is_new_format = False
            self.metadata = None
            # 旧格式：节点 ID 是字符串，作为字典的 key
            self._nodes_map: dict[int | str, dict[str, Any]] = {}
            for key, value in self.workflow.items():
                if isinstance(value, dict) and "class_type" in value:
                    # 尝试解析为整数
                    try:
                        self._nodes_map[int(key)] = value
                    except ValueError:
                        self._nodes_map[key] = value
            
            logger.info(f"Detected old format workflow with {len(self._nodes_map)} nodes")
    
    def _validate_workflow(self):
        """验证工作流结构"""
        if not isinstance(self.workflow, dict):
            raise ValueError("Workflow must be a JSON object")
        
        if not self.workflow:
            raise ValueError("Workflow is empty")
        
        # 验证输入节点是否存在
        for node_desc in self.input_node_descriptions:
            node_id = node_desc.node_id
            normalized_id = int(node_id) if isinstance(node_id, str) and node_id.isdigit() else node_id
            
            if normalized_id not in self._nodes_map:
                raise ValueError(f"Input node {node_id} not found in workflow. Available nodes: {list(self._nodes_map.keys())}")
            
            node = self._nodes_map[normalized_id]
            if "inputs" not in node:
                node["inputs"] = {}
        
        # 如果未指定输出节点，尝试自动检测
        if not self.output_node_ids:
            self.output_node_ids = self._detect_output_nodes()
            logger.info(f"Auto-detected output nodes: {self.output_node_ids}")
    
    def _detect_output_nodes(self) -> list[int | str]:
        """自动检测输出节点"""
        output_nodes = []
        for node_id, node in self._nodes_map.items():
            class_type = node.get("class_type", "")
            if class_type in self.IMAGE_OUTPUT_TYPES or class_type in self.VIDEO_OUTPUT_TYPES:
                output_nodes.append(node_id)
        return output_nodes
    
    async def run(
        self,
        prompt: Optional[str] = None,
        reference_image_paths: Optional[List[str]] = None,
        frames: Optional[List[str]] = None,
        **kwargs,
    ) -> dict[int | str, Any]:
        """
        执行工作流
        
        Args:
            prompt: 文本 prompt
            reference_image_paths: 参考图片路径列表
            frames: 帧图片路径列表（用于视频生成）
            **kwargs: 其他参数（seed, steps, cfg 等）
        
        Returns:
            执行结果字典，包含输出文件路径
        """
        # 准备输入数据
        input_data = self._prepare_inputs(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            frames=frames,
            **kwargs,
        )
        
        # 应用输入到工作流
        workflow_with_inputs = self._apply_inputs(input_data)
        
        # 应用速率限制
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        # 执行工作流
        result = await self._execute_workflow(workflow_with_inputs)
        
        return result
    
    def _prepare_inputs(
        self,
        prompt: Optional[str],
        reference_image_paths: Optional[List[str]],
        frames: Optional[List[str]],
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """准备输入数据"""
        inputs = {}
        
        for node_desc in self.input_node_descriptions:
            source = node_desc.source
            node_id = node_desc.node_id
            field_name = node_desc.field
            
            if source == NodeInputSource.PROMPT:
                inputs[f"{node_id}:{field_name}"] = prompt or node_desc.default or ""
                
            elif source == NodeInputSource.REFERENCE_IMAGE:
                if reference_image_paths:
                    images = reference_image_paths[:node_desc.max_count] if node_desc.max_count else reference_image_paths
                    # 转换为 base64 或保持路径格式（取决于工作流要求）
                    inputs[f"{node_id}:{field_name}"] = [
                        image_path_to_b64(img, mime=True) for img in images
                    ]
                else:
                    inputs[f"{node_id}:{field_name}"] = node_desc.default or []
                    
            elif source == NodeInputSource.FRAME:
                if frames and node_desc.index is not None and node_desc.index < len(frames):
                    frame_path = frames[node_desc.index]
                    inputs[f"{node_id}:{field_name}"] = image_path_to_b64(frame_path, mime=True)
                else:
                    inputs[f"{node_id}:{field_name}"] = node_desc.default
                    
            elif source == NodeInputSource.SEED:
                inputs[f"{node_id}:{field_name}"] = kwargs.get("seed", node_desc.default or 0)
                
            elif source == NodeInputSource.STEPS:
                inputs[f"{node_id}:{field_name}"] = kwargs.get("steps", node_desc.default or 20)
                
            elif source == NodeInputSource.CFG:
                inputs[f"{node_id}:{field_name}"] = kwargs.get("cfg", node_desc.default or 7.0)
                
            elif source == NodeInputSource.WIDTH:
                inputs[f"{node_id}:{field_name}"] = kwargs.get("width", node_desc.default or 1024)
                
            elif source == NodeInputSource.HEIGHT:
                inputs[f"{node_id}:{field_name}"] = kwargs.get("height", node_desc.default or 1024)
        
        return inputs
    
    def _apply_inputs(self, inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        将输入数据应用到工作流
        
        根据格式不同，处理方式也不同：
        - 新格式: 节点在 nodes 数组中，需要遍历更新
        - 旧格式: 节点直接作为字典 key
        """
        workflow = json.loads(json.dumps(self.workflow))  # 深拷贝
        
        for key, value in inputs.items():
            if ":" in key:
                node_id_str, field_name = key.split(":", 1)
                
                # 尝试将 node_id 转换为整数
                try:
                    node_id = int(node_id_str)
                except ValueError:
                    node_id = node_id_str
                
                if self.is_new_format:
                    # 新格式：在 nodes 数组中查找并更新节点
                    for node in workflow.get("nodes", []):
                        if node.get("id") == node_id:
                            if "inputs" not in node:
                                node["inputs"] = {}
                            node["inputs"][field_name] = value
                            break
                else:
                    # 旧格式：直接作为字典 key
                    if node_id in workflow and "inputs" in workflow[node_id]:
                        workflow[node_id]["inputs"][field_name] = value
        
        return workflow
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> dict[int | str, Any]:
        """执行工作流"""
        prompt_id = await self._queue_prompt(workflow)
        logger.info(f"Queued prompt: {prompt_id}")
        
        # 等待执行完成
        outputs = await self._wait_for_outputs(prompt_id)
        
        return outputs
    
    async def _queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """将工作流加入队列"""
        url = f"{self.base_url}/api/prompt"
        
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
            try:
                normalized_output_ids.add(int(oid))
            except (ValueError, TypeError):
                normalized_output_ids.add(oid)
        
        def ws_receiver():
            try:
                with ws_connect(ws_url) as ws:
                    while not result_holder["finished"]:
                        message = ws.recv()
                        data = json.loads(message)
                        
                        # 处理执行进度
                        if data.get("type") == "progress":
                            logger.debug(f"Progress: {data.get('data', {})}")
                        
                        # 处理执行完成
                        elif data.get("type") == "executed":
                            raw_node_id = data.get("data", {}).get("node", {})
                            # 尝试转换为整数（新格式）
                            try:
                                node_id = int(raw_node_id)
                            except (ValueError, TypeError):
                                node_id = raw_node_id
                            outputs[node_id] = data.get("data", {})
                            
                        # 处理错误
                        elif data.get("type") == "error":
                            result_holder["error"] = data.get("data", {})
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
        
        for node_id, output in outputs.items():
            if not output:
                continue
                
            # 检查输出内容
            if "images" in output:
                paths[node_id] = output["images"]
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
