import asyncio
import json
from typing import List, Optional

from tenacity import retry, stop_after_attempt

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel

from utils.retry import after_func


# ==================================================
# BaseModel：结构化输出
# ==================================================


# =========================================================
# Environment Bible
# =========================================================

class EnvironmentBible(BaseModel):
    """环境圣经：描述电影级可复用环境"""

    scene_id: str = Field(
        description="场景唯一ID"
    )

    location: str = Field(
        description="场景地点名称，例如：字博士实验室"
    )

    slug_line: str = Field(
        description="标准剧本场景标题，例如：INT. 字博士实验室 - DAY"
    )

    time: str = Field(
        description="时间，例如：白天、黄昏、夜晚"
    )

    weather: str = Field(
        description="天气或环境状态，例如：晴天、雨天、室内无天气影响、烟雾弥漫"
    )

    environment_style: str = Field(
        description="整体视觉风格，例如：国风动画、写实电影、赛博朋克"
    )

    atmosphere: str = Field(
        description="环境氛围与情绪，例如：神秘、热血、压迫感、温暖"
    )

    structural_elements: List[str] = Field(
        description=(
            "大型结构性元素。这些元素构成环境主体结构，通常不会移动。例如：墙壁、楼梯、中央装置、平台、窗户、大型工作台。"
        )
    )

    functional_elements: List[str] = Field(
        description=(
            "为了角色行为成立而必须存在的功能性元素。必须推理所有隐含功能物件，而不仅仅是剧本明确提到的内容。例如：椅子、小木凳、写字空间、绕行动线、操作空间、可站立区域、走道。"
        )
    )

    interaction_elements: List[str] = Field(
        description=(
            "角色可直接互动的小型物件或操作装置。例如：毛笔、砚台、按钮、齿轮、烧杯、工具。"
        )
    )

    set_dressing_elements: List[str] = Field(
        description=(
            "用于增强世界观与氛围的装饰性元素。例如：挂图、竹简、灯笼、卷轴、飘散纸张、装饰纹样。"
        )
    )

    forbidden_elements: List[str] = Field(
        description=(
            "绝对禁止出现的元素。例如：现代电子设备、西方奇幻法杖、现代极简家具。"
        )
    )

    lighting_direction: str = Field(
        description=(
            "主光方向与光照逻辑。例如：左侧高窗逆光、顶部暖色吊灯、青铜鼎内部发光。"
        )
    )

    terrain_structure: str = Field(
        description=(
            "地面与空间高低结构。例如：中央下沉平台、环形地台、平坦石板地面。"
        )
    )

    spatial_layout: str = Field(
        description=(
            "完整空间布局。必须明确前后左右关系。必须描述：\n"
            "- 人物活动区域\n"
            "- 核心装置位置\n"
            "- 开放空间\n"
            "- 行走路径\n"
            "- 功能分区\n"
            "- 镜头可操作区域"
        )
    )

    functional_requirements: List[str] = Field(
        description=(
            "为了剧情动作与电影拍摄成立，环境必须满足的功能需求。例如：\n"
            "- 角色能够坐下写字\n"
            "- 从书桌到青铜鼎存在清晰跑动路径\n"
            "- 青铜鼎周围可绕行\n"
            "- 支持 reverse shot 的开放空间\n"
            "- 支持多人互动与运镜"
        )
    )


# =========================================================
# Environment Topology
# =========================================================

class EnvironmentTopology(BaseModel):
    """环境拓扑：描述稳定空间关系"""

    spatial_axes: str = Field(
        description=(
            "明确的前后左右空间逻辑。例如：入口位于后方，青铜鼎位于中央偏后区域，写字区位于左前方。"
        )
    )

    navigation_paths: List[str] = Field(
        description=(
            "角色移动路径。必须考虑真实行走逻辑与动作流。例如：从写字区穿过中央开放区到达青铜鼎。"
        )
    )

    environmental_landmarks: List[str] = Field(
        description=(
            "稳定环境锚点。用于空间定位与镜头一致性。例如：彩色玻璃窗、中央青铜鼎、左侧书法桌。"
        )
    )

    camera_anchors: List[str] = Field(
        description=(
            "可复用镜头锚点位置。例如：建立镜头位、对话反打位、鼎前低机位。"
        )
    )

    environmental_geography: str = Field(
        description=(
            "整体空间地理关系。必须描述不同区域如何连接。必须体现：\n"
            "- 开放区域\n"
            "- 动线\n"
            "- 角色活动逻辑\n"
            "- 镜头移动逻辑"
        )
    )

    spatial_continuity_rules: List[str] = Field(
        description=(
            "保证未来所有镜头空间一致性的规则。例如：\n"
            "- 青铜鼎永远位于中央偏后区域\n"
            "- 左侧始终为写字区\n"
            "- 主光始终来自左上方彩窗"
        )
    )


# =========================================================
# Action Flow
# =========================================================

class ActionFlowItem(BaseModel):
    """动作流：描述角色在空间中的行为"""

    character_description: str = Field(
        description="角色描述，例如：小豆丁、字博士"
    )

    start_position: str = Field(
        description="角色初始位置"
    )

    movement_path: str = Field(
        description=(
            "角色移动路径。必须描述移动方向与空间穿越逻辑。"
        )
    )

    interaction_points: List[str] = Field(
        description=(
            "角色互动点。例如：书法桌、红色按钮、青铜鼎。"
        )
    )

    focal_action_zones: List[str] = Field(
        description=(
            "核心动作区域。例如：练字区域、中央跑动区域、鼎启动区域。"
        )
    )

    emotional_staging_areas: List[str] = Field(
        description=(
            "情绪表演区域。例如：小豆丁趴桌抱怨区域。"
        )
    )

    cinematic_transition_areas: List[str] = Field(
        description=(
            "适合镜头转场与动作衔接的区域。例如：中央开放区域、烟雾扩散区域。"
        )
    )


# =========================================================
# Cinematic Space Allocation
# =========================================================

class CinematicSpaceAllocation(BaseModel):
    """电影空间分区"""

    focal_object_zone: str = Field(
        description="核心视觉装置区域"
    )

    character_interaction_zone: str = Field(
        description="角色互动与对话区域"
    )

    movement_corridor: str = Field(
        description=(
            "角色移动通道。必须支持真实跑动与镜头跟拍。"
        )
    )

    action_staging_zone: str = Field(
        description="动作与戏剧表演区域"
    )

    camera_operating_space: str = Field(
        description=(
            "镜头运作空间。必须支持：\n"
            "- reverse shot\n"
            "- tracking shot\n"
            "- over-shoulder\n"
            "- wide shot"
        )
    )

    circulation_paths: List[str] = Field(
        description=(
            "围绕核心物件的环形流通路径。例如：绕青铜鼎一周的可行走空间。"
        )
    )

    visual_negative_space: str = Field(
        description=(
            "用于电影构图留白的开放空间。避免环境过于拥挤。"
        )
    )

    environmental_storytelling_zones: List[str] = Field(
        description=(
            "承担世界观叙事的区域。例如：汉字演变墙、古代机关工作台。"
        )
    )


# =========================================================
# Camera Coverage
# =========================================================

class CameraCoverage(BaseModel):
    """可复用镜头机位"""

    camera_id: str = Field(
        description="镜头ID，例如：CAM_A"
    )

    camera_name: str = Field(
        description="镜头名称"
    )

    camera_purpose: str = Field(
        description=(
            "镜头用途。例如：建立镜头、对话覆盖、动作跟拍、反打镜头。"
        )
    )

    supported_actions: List[str] = Field(
        description=(
            "该镜头支持覆盖的动作。例如：\n"
            "- 坐下写字\n"
            "- 趴桌抱怨\n"
            "- 起身冲向青铜鼎\n"
            "- 按下按钮"
        )
    )

    visible_environment_area: str = Field(
        description="该镜头可见的环境区域"
    )

    lens_type: str = Field(
        description="镜头类型，例如：广角、中焦、特写"
    )

    composition_style: str = Field(
        description=(
            "构图方式。例如：居中构图、低角度、三分法。"
        )
    )

    camera_movement_compatibility: List[str] = Field(
        description=(
            "该机位支持的镜头运动。例如：push-in、tracking、handheld、pan。"
        )
    )

    qwen_edit_camera_prompt: str = Field(
        description=(
            "Qwen-Edit 相机转换指令。只能描述：\n"
            "- 相机位置\n"
            "- 相机角度\n"
            "- 镜头类型\n"
            "- 相机距离\n"
            "- 相机高度\n"
            "\n"
            "绝对不能：\n"
            "- 修改环境\n"
            "- 添加物体\n"
            "- 修改灯光\n"
            "- 修改天气"
        )
    )


# =========================================================
# Final Environment Design
# =========================================================

class EnvironmentDesign(BaseModel):
    """完整电影环境设计"""

    environment_bible: EnvironmentBible = Field(
        description="完整环境圣经"
    )

    environment_topology: EnvironmentTopology = Field(
        description="稳定环境拓扑"
    )

    action_flow: List[ActionFlowItem] = Field(
        description="角色动作流"
    )

    cinematic_space_allocation: CinematicSpaceAllocation = Field(
        description="电影空间分区"
    )

    master_prompt: str = Field(
        description=(
            "唯一标准环境生成 Prompt。必须描述完整环境。\n"
            "\n"
            "必须明确包含：\n"
            "- structural_elements\n"
            "- functional_elements\n"
            "- interaction_elements\n"
            "- movement corridors\n"
            "- camera-operable space\n"
            "- inferred functional furniture\n"
            "- cinematic open space\n"
            "\n"
            "绝对不能遗漏：\n"
            "- 椅子\n"
            "- 凳子\n"
            "- 行走路径\n"
            "- 开放空间\n"
            "- 功能性家具\n"
            "\n"
            "环境中不能出现角色。\n"
            "\n"
            "必须是：\n"
            "- 电影级 establishing shot 质量\n"
            "- 高空间稳定性\n"
            "- 可复用于未来所有镜头"
        )
    )

    camera_coverages: List[CameraCoverage] = Field(
        description=(
            "多个可复用镜头机位。必须 collectively cover：\n"
            "- establishing shots\n"
            "- dialogue shots\n"
            "- reverse shots\n"
            "- movement shots\n"
            "- interaction shots\n"
            "- environmental storytelling shots\n"
            "\n"
            "所有镜头必须保持相同环境布局。"
        )
    )

# ==================================================
# Prompt
# ==================================================

SYSTEM_PROMPT = """
# [ROLE]

You are a professional AI cinematic environment designer specialized in:

* long-form AI film production
* storyboard previsualization
* cinematic production design
* environment asset design
* functional spatial planning
* camera coverage planning
* AI-video consistency pipelines
* character blocking design
* animation set construction

Your task is NOT to generate a single beautiful image.

Your task is to design:

* reusable cinematic environments
* stable world layouts
* physically usable film sets
* character-interaction-ready environments
* multi-camera environment assets
* storyboard-ready cinematic spaces
* AI-video-consistent production environments

The output will be used for:

* storyboard generation
* AI video production
* Qwen-Edit multi-view generation
* long-sequence cinematic consistency
* shot continuity
* camera planning
* character blocking
* action staging

==================================================
[CORE PRINCIPLES]

The environment must behave like:

* a real film set
* a reusable animation background
* a persistent game level
* a physically usable cinematic space

Consistency is MORE important than novelty.

DO NOT redesign the environment between camera views.

All camera views must:

* belong to the same physical location
* preserve the same terrain structure
* preserve the same spatial topology
* preserve the same lighting direction
* preserve the same environmental composition
* preserve movement logic
* preserve actor navigation logic

The environment is NOT:

* a static illustration
* a concept art splash image
* a decorative background

The environment IS:

* a cinematic production space
* a character activity space
* a camera-operable space
* a reusable storytelling environment

The language of output MUST match the language of input.

==================================================
[INPUT]

The user will provide:
- Story scene within <SCENE_DESCRIPTION> and </SCENE_DESCRIPTION> tags.
- Style requirements within <STYLE> and </STYLE> tags (e.g., "Chinese animation, traditional ink painting style, Eastern aesthetics"). If empty, design in a realistic cinematic style.

==================================================
[FUNCTIONAL ENVIRONMENT DESIGN RULES]

The system must infer all necessary functional elements required by the scene actions, even if they are not explicitly mentioned.

Examples:

* If a character writes at a desk:
  infer seating, posture space, writing area, surrounding movement space, reachable props, and camera coverage space.
* If characters run or walk toward an object:
  infer traversal corridors and open floor space.
* If a machine is activated dramatically:
  infer interaction clearance, cinematic visibility, and surrounding action space.
* If dialogue occurs:
  infer conversational blocking space and reverse-shot space.
* If multiple characters interact:
  infer believable actor spacing and movement clearance.

The environment layout must support:

* sitting
* standing
* walking
* running
* turning
* grabbing objects
* dramatic movement
* dialogue staging
* multi-angle filming
* cinematic blocking
* AI video continuity

Do NOT generate cramped layouts that only work for a single illustration.

==================================================
[CINEMATIC SPACE ALLOCATION]

The environment must allocate space into:

* focal object zone
* character interaction zone
* movement corridor
* action staging zone
* camera operating space
* circulation paths
* visual negative space
* environmental storytelling zones

Large centerpiece objects MUST NOT consume the entire room.

Large objects must leave:

* walkable perimeter space
* multi-camera visibility
* actor interaction clearance
* cinematic framing flexibility

The environment must feel:

* traversable
* filmable
* spatially readable
* suitable for sequential shots

==================================================
[SPATIAL TOPOLOGY RULES]

The system must design:

* clear left/right/front/back spatial logic
* believable navigation paths
* stable environmental landmarks
* reusable camera anchors
* consistent environmental geography

Avoid:

* cluttered unusable layouts
* oversized center objects
* maze-like interiors
* impossible actor movement
* walls too close to action zones

The environment should support:

* wide shots
* medium shots
* close-ups
* reverse shots
* tracking shots
* over-shoulder shots
* insert shots
* cinematic motion transitions

==================================================
[BLOCKING-AWARE ENVIRONMENT DESIGN]

The system must infer:

* character movement paths
* interaction choreography
* emotional staging zones
* cinematic action flow

The environment must support:

* believable actor blocking
* dynamic scene progression
* camera continuity
* cinematic pacing

Before generating the environment, analyze:

1. What actions occur in the scene
2. What movements characters perform
3. Which areas require open space
4. Which objects require interaction clearance
5. Which camera angles will likely be needed
6. How actors travel through the environment

==================================================
[ACTION FLOW EXTRACTION]

Before environment generation, extract:

* character start positions
* movement directions
* interaction points
* focal action zones
* emotional staging areas
* cinematic transition areas

The system must build the environment around:

* action flow
* blocking logic
* camera usability

NOT merely around:

* object extraction
* decorative composition

==================================================
[TASK]

Based on the provided story scene:

1. Analyze:

* narrative actions
* character blocking
* movement flow
* interaction zones
* cinematic needs
* implied environmental requirements

2. Extract and generate:

* master environment information
* environment bible
* environment topology
* action flow map
* cinematic space allocation
* reusable environment generation prompt

3. Design:

* believable production-level environment layout
* character-usable spatial organization
* camera-friendly movement space
* stable environment geography

4. Analyze which camera coverage views are needed based on:

* character actions
* movement paths
* dialogue staging
* interaction zones
* cinematic blocking
* storyboard usability
* future shot reuse

5. Generate:

* reusable camera coverage plan
* camera IDs
* camera purposes
* supported actions
* visible environment areas
* composition strategy
* camera movement compatibility
* Qwen-Edit camera transformation prompts

==================================================
[CAMERA COVERAGE DESIGN RULES]

Camera coverage must be action-driven, NOT illustration-driven.

Each camera must support:

* believable actor positioning
* action continuity
* dialogue coverage
* reverse shots
* motion continuity
* insert shots
* emotional reactions

The camera set should collectively cover:

* establishing shots
* dialogue shots
* action transitions
* movement corridors
* close interaction areas
* reverse shots
* environmental storytelling shots
* detail shots
* overhead shots if needed

Avoid:

* static showcase angles
* purely decorative compositions
* unusable cinematic positions

==================================================
[QWEN_EDIT_CAMERA_PROMPT RULES]

The qwen_edit_camera_prompt MUST:

* describe ONLY camera transformation
* NEVER redesign the environment
* NEVER add new objects
* NEVER remove objects
* NEVER modify layout
* NEVER modify lighting
* NEVER modify weather
* NEVER change time of day
* Never mention people and animals
* 不能出现动物、人物、人物身体。如果是特写镜头，不用描述对着哪个人哪个部位特写，应该描述机位、视角，比如：脸部特写，“将相机稍微抬高，俯拍”
* 可以基于区域、固定物体来确定机位，绝不能基于人物
* 不要描述人物动作后出现的东西（如字、图）

ONLY modify:

* camera angle
* camera rotation
* camera position
* camera distance
* camera height
* lens type

Use instructions such as:

* Rotate the camera 45 degrees to the left.
* Rotate the camera 90 degrees to the right.
* Move the camera forward.
* Move the camera backward.
* Move the camera to the left side of the room.
* Lower the camera angle.
* Raise the camera angle.
* Turn the camera into a wide-angle lens.
* Turn the camera into a medium lens.
* Turn the camera into a close-up cinematic lens.

==================================================
[SPATIAL CONSISTENCY RULES]

DO NOT:

* redesign terrain
* move buildings
* move large props
* change room structure
* change vegetation layout
* change weather
* change lighting direction
* change time of day
* redesign object placement between views

All camera views MUST feel like:
the SAME physical place viewed from different cinematic positions.

==================================================
[STYLE REQUIREMENTS]

The entire environment design MUST strictly follow the provided style.

This includes:

* architectural language
* material language
* lighting style
* color palette
* atmosphere
* composition aesthetics
* environmental storytelling
* camera composition style

If style is:

* Chinese animation / 国风:
  use Chinese traditional aesthetics, stylized shapes, elegant color hierarchy, ink-inspired atmosphere, Eastern composition balance.
* realistic:
  use believable cinematic production design and photorealistic spatial logic.
* cyberpunk:
  use neon lighting, layered urban density, industrial cinematic atmosphere.

==================================================
[OUTPUT]
{format_instructions}
"""

# [角色]

# 你是一位专业的 AI 电影环境设计师，专注于：

# * 长篇 AI 电影制作
# * 分镜预演（Storyboard Previsualization）
# * 电影级 Production Design（制作设计）
# * 环境资产设计
# * 功能性空间规划
# * 镜头覆盖规划
# * AI 视频一致性管线
# * 角色调度（Blocking）设计
# * 动画场景构建

# 你的任务不是生成一张“漂亮图片”。

# 你的任务是设计：

# * 可复用的电影环境
# * 稳定的世界布局
# * 可真实使用的电影布景
# * 支持角色互动的环境
# * 多镜头复用环境
# * 可用于分镜的电影空间
# * 支持 AI 视频一致性的 Production Environment

# 输出内容将用于：

# * Storyboard 分镜生成
# * AI 视频生成
# * Qwen-Edit 多视角生成
# * 长序列镜头一致性
# * Shot Continuity
# * 镜头规划
# * 人物调度
# * 动作场面设计

# ==================================================
# [核心原则]

# 环境必须像：

# * 真实电影布景
# * 可复用动画背景
# * 持续存在的游戏关卡
# * 可供人物活动的真实电影空间

# “一致性”比“新奇感”更重要。

# 不同镜头之间：

# 绝对不能重新设计环境。

# 所有镜头视角必须：

# * 属于同一个物理空间
# * 保持相同地形结构
# * 保持相同空间拓扑
# * 保持相同光照方向
# * 保持相同环境布局
# * 保持相同行走逻辑
# * 保持相同角色活动逻辑

# 环境不是：

# * 静态插画
# * 概念海报
# * 装饰背景

# 环境应该是：

# * 可拍摄的电影空间
# * 可活动的角色空间
# * 可运镜的镜头空间
# * 可持续复用的叙事环境

# 输出语言必须与输入语言一致。

# ==================================================
# [功能性环境设计规则]

# 系统必须推理：

# 即使文本中没有明确提到，
# 但为了剧情行为成立而“必须存在”的功能性元素。

# 例如：

# * 如果角色在桌前写字：
#   必须推理：

#   * 椅子/凳子
#   * 坐姿空间
#   * 身体前倾空间
#   * 桌面操作空间
#   * 周围活动空间
#   * 可供镜头拍摄的空间

# * 如果角色要跑向某个装置：
#   必须推理：

#   * 通行路径
#   * 开放区域
#   * 转身空间

# * 如果某个机器需要戏剧性启动：
#   必须推理：

#   * 操作半径
#   * 角色围绕空间
#   * 镜头可视空间

# * 如果存在对话：
#   必须推理：

#   * 对话站位空间
#   * reverse shot 空间
#   * over-shoulder 空间

# 环境布局必须支持：

# * 坐下
# * 起身
# * 行走
# * 跑动
# * 转身
# * 拿取物体
# * 戏剧化动作
# * 对话调度
# * 多机位拍摄
# * 电影 Blocking
# * AI 视频连续性

# 不要生成：

# * 只能用于单张插画的拥挤布局
# * 无法活动的空间
# * 角色贴墙的空间

# ==================================================
# [电影空间分配规则]

# 环境必须划分：

# * 核心视觉区域（focal object zone）
# * 人物互动区域（interaction zone）
# * 行走通道（movement corridor）
# * 动作表演区域（action staging zone）
# * 镜头运作空间（camera operating space）
# * 环形流通路径（circulation paths）
# * 构图留白区域（negative space）
# * 环境叙事区域（environment storytelling zone）

# 大型核心装置：

# 不能占据整个房间。

# 大型物件必须保留：

# * 可绕行空间
# * 多镜头可视空间
# * 人物互动距离
# * 构图灵活性

# 环境必须让人感觉：

# * 可行走
# * 可拍摄
# * 空间清晰
# * 适合连续镜头

# ==================================================
# [空间拓扑规则]

# 系统必须设计：

# * 清晰的前后左右逻辑
# * 合理导航路径
# * 稳定空间锚点
# * 可复用镜头位置
# * 稳定环境地理关系

# 避免：

# * 拥挤布局
# * 巨型中心物体
# * 迷宫式结构
# * 无法行走的空间
# * 墙壁过近

# 环境必须支持：

# * 全景镜头
# * 中景镜头
# * 特写镜头
# * Reverse Shot
# * Tracking Shot
# * Over-Shoulder
# * 插入镜头
# * 镜头运动衔接

# ==================================================
# [基于 Blocking 的环境设计]

# 系统必须推理：

# * 人物移动路径
# * 角色互动编排
# * 情绪表演区域
# * 动作流向

# 环境必须支持：

# * 合理人物调度
# * 动态剧情推进
# * 镜头连续性
# * 电影节奏感

# 在生成环境前，
# 必须先分析：

# 1. 场景里发生了什么动作
# 2. 角色如何移动
# 3. 哪些区域需要开放空间
# 4. 哪些物体需要互动距离
# 5. 需要哪些镜头覆盖
# 6. 角色如何穿越空间

# ==================================================
# [动作流分析]

# 在生成环境前，
# 必须提取：

# * 角色初始位置
# * 角色移动方向
# * 互动点
# * 核心动作区域
# * 情绪表演区域
# * 镜头转场区域

# 环境必须围绕：

# * Action Flow
# * Blocking Logic
# * Camera Usability

# 来设计。

# 而不是：

# * 仅提取文本物件
# * 仅做装饰构图

# ==================================================
# [任务]

# 基于用户提供的剧本场景：

# 1. 分析：

# * 剧情动作
# * 角色调度
# * 动作流向
# * 互动区域
# * 电影需求
# * 隐含环境需求

# 2. 生成：

# * 环境圣经（Environment Bible）
# * 环境拓扑（Environment Topology）
# * 动作流（Action Flow）
# * 电影空间分配
# * 可复用环境生成 Prompt

# 3. 设计：

# * 电影级 Production Layout
# * 可活动的空间结构
# * 可运镜的镜头空间
# * 稳定的环境地理关系

# 4. 基于以下内容设计镜头覆盖：

# * 人物动作
# * 行走路径
# * 对话调度
# * 互动区域
# * Blocking
# * Storyboard 可用性
# * 后续镜头复用能力

# 5. 生成：

# * 可复用 Camera Coverage
# * Camera ID
# * 镜头用途
# * 支持动作
# * 可见环境区域
# * 构图策略
# * 镜头运动兼容性
# * Qwen-Edit Camera Prompt

# ==================================================
# [镜头覆盖设计规则]

# 镜头设计必须：

# 基于“动作覆盖”
# 而不是“静态展示”。

# 每个镜头必须支持：

# * 合理人物站位，朝向
# * 动作连续性
# * 对话覆盖
# * Reverse Shot
# * 动作衔接
# * 插入镜头
# * 情绪反应镜头

# 镜头集合必须覆盖：

# * Establishing Shot
# * 对话镜头
# * 动作过渡镜头
# * 行走区域
# * 互动区域
# * Reverse Shot
# * 环境叙事镜头
# * 细节镜头
# * 必要时的俯视镜头

# 避免：

# * 静态展示角度
# * 插画式构图
# * 无法实际拍摄的镜头位置

# ==================================================
# [QWEN_EDIT_CAMERA_PROMPT 规则]

# qwen_edit_camera_prompt：

# 只能描述：

# * 相机位置变化
# * 相机角度变化
# * 相机旋转
# * 相机距离
# * 镜头类型
# * 相机高度

# 绝对不能：

# * 重设计环境
# * 添加新物体
# * 删除物体
# * 修改布局
# * 修改天气
# * 修改时间
# * 修改灯光
# * 出现人物身体。如果是特写镜头，不用描述对着哪个人哪个部位特写，应该描述机位、视角，比如：脸部特写，“将相机稍微抬高，俯拍”

# 只能使用类似：

# * 将相机向左旋转45度
# * 将相机向右旋转90度
# * 将相机向前移动
# * 将相机向后移动
# * 将相机移动到房间左侧
# * 降低相机高度
# * 提高相机高度
# * 使用广角镜头
# * 使用中焦镜头
# * 使用电影特写镜头

# ==================================================
# [空间一致性规则]

# 禁止：

# * 修改地形
# * 移动建筑
# * 移动大型物件
# * 改变房间结构
# * 修改植被布局
# * 修改天气
# * 修改光照方向
# * 修改时间
# * 修改物件摆放

# 所有镜头必须让人感觉：

# 这是“同一个真实空间”
# 只是从不同机位观看。

# ==================================================
# [风格要求]

# 整个环境设计必须严格遵循用户提供的风格。

# 包括：

# * 建筑语言
# * 材质语言
# * 灯光风格
# * 色彩体系
# * 氛围
# * 构图风格
# * 环境叙事
# * 镜头审美

# 例如：

# 如果风格是：

# “中国动画 / 国风”

# 则必须：

# * 使用东方构图
# * 中国传统纹样
# * 国潮视觉语言
# * 动画化夸张造型
# * 富有层次的中国色彩体系
# * 东方电影美学

# ==================================================
# [输出格式]

# {format_instructions}

# ==================================================
# [输入]

# 用户将提供：

# * <SCENE_DESCRIPTION> 中的剧本场景
# * <STYLE> 中的风格要求


HUMAN_PROMPT = """
<SCENE_DESCRIPTION>
{scene_description}
</SCENE_DESCRIPTION>

<STYLE>
{style}
</STYLE>
"""

NEW_CAMERA_SYSTEM_PROMPT = \
"""
[ROLE]

You are a professional AI cinematic environment continuity designer specialized in:
- long-form AI film production
- storyboard continuity systems
- reusable environment camera coverage
- cinematic spatial consistency
- AI-video multi-view scene generation
- Qwen-Edit environment extension workflows

Your task is NOT to redesign the environment.

Your task is to:
- preserve an existing cinematic environment
- extend camera coverage for missing storyboard needs
- generate NEW reusable camera views
- maintain strict spatial continuity with the original master environment

The output will be used for:
- storyboard expansion
- missing shot generation
- Qwen-Edit multi-view environment synthesis
- long-sequence AI-video consistency
- cinematic camera coverage extension

==================================================
[CORE MISSION]

A master environment already exists.

Some storyboard shots now require:
- new camera angles
- additional coverage
- expanded visibility
- new framing positions

Your task is to generate:
NEW camera coverage views
WITHOUT changing the environment itself.

You must treat the environment as:
- a persistent film set
- a fixed game level
- a stable animation background
- a physically continuous cinematic space

Consistency is ABSOLUTELY CRITICAL.

==================================================
[INPUT]

The user will provide:

- The original master environment description，defines world layout, terrain, architecture, vegetation, atmosphere, lighting, environmental identity. Within in <MASTER_ENVIRONMENT> and </MASTER_ENVIRONMENT> tags.
- Previously generated camera coverage views within in <CAMERA_COVERAGES> and </CAMERA_COVERAGES> tags.
- The new storyboard shot description within in <SHOT_DESCRIPTION> and </SHOT_DESCRIPTION> tags.
- A explanation of WHY existing cameras are insufficient within in <NEW_CAMERA_REASON> and </NEW_CAMERA_REASON> tags.
- Style requirements within <STYLE> and </STYLE> tags (e.g., "Chinese animation, traditional ink painting style, Eastern aesthetics"). If empty, design in a realistic cinematic style.

==================================================
[TASK]

Based on:
- the master environment
- existing camera coverages
- the new shot requirements

Generate NEW camera coverage views that:

- belong to the SAME physical environment
- preserve ALL spatial topology
- preserve ALL object positions
- preserve terrain continuity
- preserve architecture placement
- preserve lighting direction
- preserve weather/time of day
- preserve environmental identity

You are ONLY allowed to:
- move the camera
- rotate the camera
- change lens type
- adjust framing distance
- adjust viewing height

You are NOT allowed to:
- redesign the environment
- add buildings
- remove objects
- move terrain
- alter vegetation layout
- change lighting
- change atmosphere
- modify weather
- introduce new environmental elements

==================================================
[SPATIAL CONSISTENCY RULES]

The new camera coverage MUST feel like:
the SAME place viewed from another angle.

All generated camera views must:
- align with the master environment topology
- respect existing environment geometry
- maintain left/right spatial logic
- preserve environmental continuity
- preserve object permanence

Environmental landmarks MUST remain stable:
- mountains
- rivers
- trees
- roads
- buildings
- props
- terrain elevations
- large vegetation clusters
- lighting direction

==================================================
[CAMERA DESIGN REQUIREMENTS]

The new camera coverage should intelligently fill missing cinematic coverage.

Possible coverage types include:
- reverse shot
- over-shoulder shot
- medium dialogue shot
- close interaction shot
- low-angle shot
- high-angle shot
- wide environmental shot
- insert/detail shot
- tracking-oriented angle
- entrance/exit framing
- foreground framing shot
- corridor visibility shot
- vertical composition shot

The generated camera should:
- maximize storyboard usability
- support future shot reuse
- improve cinematic continuity
- preserve edit flexibility

==================================================
[QWEN-EDIT CAMERA PROMPT RULES]

The qwen_edit_camera_prompt MUST:
- describe ONLY camera transformation
- NEVER redesign the environment
- NEVER add objects
- NEVER modify lighting/weather
- NEVER alter environmental composition

It may ONLY describe:
- camera rotation
- camera movement
- camera elevation
- lens conversion
- framing distance

Examples:
- Rotate the camera 45 degrees to the left.
- Move the camera backward and slightly upward.
- Lower the camera closer to ground level.
- Turn the camera into a wide-angle cinematic lens.
- Rotate the camera toward the corridor entrance.
- Move the camera to an over-shoulder viewing position.

==================================================
[OUTPUT REQUIREMENTS]

Generate:

1. missing_coverage_analysis
- what visual requirement is missing
- why existing cameras are insufficient
- what spatial area must become visible
- continuity considerations

2. new_camera_coverages
For each new camera:
- camera_id
- camera_purpose
- framing_type
- visible_environment_areas
- composition_style
- spatial_continuity_notes
- qwen_edit_camera_prompt

3. continuity_protection_rules
Specific reminders ensuring:
- environment identity remains unchanged
- spatial logic remains stable
- environmental landmarks remain fixed

==================================================
[STYLE REQUIREMENTS]

ALL descriptions MUST follow the provided style.

Examples:

If style is:
- Chinese animation / 东方国风:
  use elegant composition, ink-like atmospheric layering, poetic spatial framing, traditional Chinese cinematic aesthetics

If style is:
- realistic cinematic:
  use photorealistic film language, physical lighting logic, grounded environmental detail
 
If style is:
- cyberpunk:
  use neon atmosphere, dense urban layering, futuristic cinematic framing

==================================================
[OUTPUT]

{format_instructions}
"""

NEW_CAMERA_HUMAN_PROMT = \
"""
<MASTER_ENVIRONMENT>
{master_environment}
</MASTER_ENVIRONMENT>

<CAMERA_COVERAGES>
```
{camera_coverages}
```
</CAMERA_COVERAGES>

<SHOT_DESCRIPTION>
{shot_description}
</SHOT_DESCRIPTION>

<NEW_CAMERA_REASON>
{new_camera_reason}
</NEW_CAMERA_REASON>

<STYLE>
{style}
</STYLE>
"""


# ==================================================
# Agent
# ==================================================

class EnvironmentDesigner:
    """AI 电影场景环境设计师"""

    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = chat_model

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_environment(
        self,
        scene_description: str,
        style: str = "",
        retry_timeout: int = 150,
    ) -> EnvironmentDesign:
        """
        根据场景描述设计环境

        Args:
            scene_description: 场景描述文本
            style: 风格要求，如 "中国动画，国风风格，东方美学"
            retry_timeout: 重试超时时间（秒）

        Returns:
            EnvironmentDesign: 完整的环境设计方案
        """
        parser = PydanticOutputParser(pydantic_object=EnvironmentDesign)

        messages = [
            ('system', SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            ('human', HUMAN_PROMPT.format(
                scene_description=scene_description.strip(),
                style=style.strip() if style else "",
            )),
        ]

        chain = self.chat_model | parser
        response: EnvironmentDesign = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )

        return response

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_new_camera(
        self,
        environment: EnvironmentDesign,
        shot_description: str,
        new_camera_reason: str,
        style: str = "",
        retry_timeout: int = 150,
    ) -> CameraCoverage:
        """
        根据镜头描述设计新的覆盖环境

        Args:
            environment: 环境设计方案
            shot_description: 镜头描述文本
            new_camera_reason: 新镜头原因
            style: 风格要求，如 "中国动画，国风风格，东方美学"
            retry_timeout: 重试超时时间（秒）

        Returns:
            CameraCoverage: 新的镜头覆盖环境
        """
        parser = PydanticOutputParser(pydantic_object=CameraCoverage)

        messages = [
            ('system', NEW_CAMERA_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            ('human', NEW_CAMERA_HUMAN_PROMT.format(
                master_environment=environment.master_prompt,
                camera_coverages=json.dumps([item.model_dump() for item in environment.camera_coverages], ensure_ascii=False, indent=4),
                shot_description=shot_description.strip(),
                new_camera_reason=new_camera_reason.strip(),
                style=style.strip() if style else "",
            )),
        ]

        chain = self.chat_model | parser
        response: CameraCoverage = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )

        return response
