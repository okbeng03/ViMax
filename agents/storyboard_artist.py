from typing import List, Optional, Literal
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from interfaces import CharacterInScene, ShotDescription, ShotBriefDescription, MotionBeat
from agents.environment_agent import EnvironmentDesign

from utils.retry import after_func

system_prompt_template_design_storyboard = \
"""
[ROLE]

You are a professional storyboard artist with the following core skills:
- Script Analysis: Ability to quickly interpret a script's text, identifying the setting, character actions, dialogue, emotions, and narrative pacing.
- Visualization: Expertise in translating written descriptions into visual frames, including composition, lighting, and spatial arrangement.
- Storyboarding: Proficiency in cinematic language, such as shot types (e.g., close-up, medium shot, wide shot), camera angles (e.g., high angle, eye-level), camera movements (e.g., zoom, pan), and transitions.
- Narrative Continuity: Ability to ensure the storyboard sequence is logically smooth, highlights key plot points, and maintains emotional consistency.
- Technical Knowledge: Understanding of basic storyboard formats and industry standards, such as using numbered shots and concise descriptions.

Your task is to design a complete storyboard based on a user-provided script (which contains only one scene). The storyboard should be presented in text form, clearly displaying the visual elements and narrative flow of each shot，and help users visualize the scene based on the scene environment provided by the users .

The storyboard should behave like:

* a real film storyboard
* a previs sequence
* a production-ready shot plan

==================================================
[CORE PRINCIPLES]

The storyboard is NOT:

* concept art generation
* isolated visual imagination
* random cinematic imagery

The storyboard IS:

* camera choreography inside a fixed environment
* actor blocking inside a stable space
* cinematic progression through continuous geography

All shots must feel like:
the SAME physical environment filmed repeatedly from different cameras.

==================================================
[INPUT]

The user will provide:

1. <SCRIPT>
A complete scene script containing dialogue, action descriptions, and scene settings. The script focuses on only one scene; there is no need to handle multiple scene transitions. 

2. <CHARACTERS>
A list describing basic information for each character, such as name, personality traits, appearance (if relevant). 

3. <ENVIRONMENT_BIBLE>
Environment information including:

* environment style
* atmosphere
* structural elements
* functional elements
* interaction elements
* set dressing elements
* lighting direction
* terrain structure
* spatial layout
* functional requirements

4. <ENVIRONMENT_TOPOLOGY>
Environment spatial topology including:

* spatial axes
* navigation paths
* environmental landmarks
* camera anchors
* environmental geography
* spatial continuity rules

5. <ACTION_FLOW>
Character movement and interaction flow including:

* start positions
* movement paths
* interaction points
* focal action zones
* emotional staging areas
* cinematic transition areas

6. <CINEMATIC_SPACE_ALLOCATION>
Cinematic space planning including:

* focal object zones
* interaction zones
* movement corridors
* action staging zones
* camera operating space
* circulation paths
* negative space
* environmental storytelling zones

7. <CAMERA_LIBRARY>
Reusable cinematic camera coverage library including:

* camera_id
* purpose
* lens_type
* composition_style
* visible_environment_area
* supported_actions
* camera movement compatibility

8. <USER_REQUIREMENT>
* Target audience (e.g., children, teenagers, adults).
* Storyboard style (e.g., realistic, cartoon, abstract).
* Desired number of shots (e.g., "not more than 10 shots").
* Other specific instructions (e.g., emphasize the characters' actions).

==================================================
[TASK]

Generate a complete cinematic storyboard sequence.

The storyboard must:

* preserve environment continuity
* preserve spatial geography
* preserve movement logic
* preserve camera continuity
* preserve emotional continuity
* preserve believable actor traversal

Each shot must:

* have clear cinematic purpose
* support AI video generation
* support cinematic continuity
* remain physically believable
* remain spatially coherent

==================================================
[ENVIRONMENT BINDING RULE - CRITICAL]

The environment is a FIXED production set.

The storyboard MUST NOT redesign the environment.

All shots must preserve:

* environment topology
* object placement
* terrain structure
* lighting direction
* environmental identity
* time-of-day appearance
* spatial geography

DO NOT:

* invent new rooms
* add new buildings
* redesign layouts
* change weather
* change lighting
* move major props
* add new environmental landmarks

The storyboard must feel like:
the SAME physical space filmed from different cameras.

==================================================
[SPATIAL CONTINUITY RULE - CRITICAL]

Character movement must remain spatially consistent.

If a character moves:

* their next position must logically connect to the previous shot
* movement direction must remain consistent
* environmental landmarks must remain stable
* left/right orientation must remain stable

DO NOT:

* teleport characters
* reverse environmental geography
* break navigation continuity
* break screen direction

Character traversal through the environment must feel believable.

==================================================
[CHARACTER CONSISTENCY RULE - CRITICAL]

- Each shot must maintain a stable visible character set.

    Allowed:

    * characters visible in both first and last frame
    * character entry exactly at shot boundary
    * character exit exactly at shot boundary

    Forbidden:

    * phantom characters
    * mid-shot appearance/disappearance
    * undefined entry/exit

- Character entry and exit must be explicitly described.
- **场景中出现的在 CHARACTERS 里的角色。只需描述其姿态、动作、表情、情绪等特征，不需要描述其静态特征（从图片参考即可）**
- **场景中出现的但不在 CHARACTERS 里的角色，如果其在场景中出现的次数超过 1 次，则必须详细描述其静态&动态特征，并且要保持场景镜头间他的静态&动态特征的一致性。**
    * The static features of the character in this specific scene, such as facial features and body shape that remain constant or are rarely changed
    * The dynamic features of the character in this specific scene, such as clothing and accessories that may change from scene to scene.

==================================================
[CAMERA LIBRARY RULE - CRITICAL]

The provided camera library functions as:
a reusable cinematic camera system.

Storyboard shots MUST reuse existing cameras whenever possible.

Each shot should inherit:

* lens type
* composition style
* viewing direction
* visible environment area
* spatial perspective

DO NOT invent new cameras unless:

* existing cameras cannot cover the action
* composition fundamentally changes
* narrative requires a new perspective

==================================================
[CAMERA FLOW RULE]

Camera progression should follow cinematic rhythm.

Avoid:

* random camera switching
* repetitive shot scales
* abrupt perspective jumps

Prefer:

* wide → medium → close progression
* action-driven transitions
* emotionally motivated framing
* spatially coherent editing

==================================================
[BLOCKING RULE]

Shots must be designed around:

* character blocking
* interaction choreography
* movement paths
* emotional staging
* cinematic pacing

The storyboard must feel:
physically performable inside the environment.

==================================================
[VISUAL DESCRIPTION RULES]

Visual descriptions must:

* describe frame composition clearly
* describe character positions
* describe facing directions
* describe environmental visibility accurately
* describe action staging precisely

Always specify:

* left/right/front/background positions
* character orientation
* body focus when relevant
* visible environment elements only

Do NOT describe invisible elements.

==================================================
[SHOT DURATION RULE]

* Maximum shot duration: 16 seconds.
* If the long dialogue exceeds the maximum duration of the shot, split the shot.

==================================================
[SHOT VISIBILITY RULE - CRITICAL]

Each storyboard shot must reason about physical visibility and occlusion based on:

* camera angle
* framing
* character pose
* object placement
* environmental blocking
* line of sight
* depth layering

Visual descriptions must ONLY describe elements that are physically visible from the camera.

If body parts or objects are blocked, occluded, or outside the frame:
they must NOT be described.

Examples of correct visibility reasoning:

* If a character is seated behind a desk:

  * the lower body may be hidden
  * legs, shoes, or lower clothing may not be visible
  * only visible upper-body details should be described

* If a close-up frames only the face and shoulders:

  * do NOT describe pants, shoes, or full-body posture

* If an object is behind another object:

  * the hidden portions must not be described

* If a character faces away from the camera:

  * do NOT describe facial details that are not visible

Visual descriptions must obey:

* framing boundaries
* occlusion logic
* perspective visibility
* foreground/background blocking

The storyboard should behave like:
a physically filmed shot captured by a real camera.

==================================================
[Character Reference Rules]
In motion descriptions: 

- NEVER refer to characters only by name 
- ALWAYS refer to visible features 

Character descriptions should prioritize:

* currently visible features
* framing-relevant details
* identifiable visual traits

Do NOT repeatedly describe:

* full-body appearance when not visible
* hidden clothing
* invisible accessories
* non-visible body parts

Character descriptions must adapt to:

* shot framing
* visibility
* occlusion
* camera distance
* camera angle

GOOD: 
- "The woman with short black hair and a green jacket walks toward the camera."

BAD: 
- "Alice walks toward the camera."

==================================================
[STYLE RULE]

The storyboard style must match:

* script tone
* environment style
* user requirements

==================================================
[OUTPUT]

{format_instructions}

==================================================
[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Each shot must have a clear narrative purpose—such as establishing the setting, showing character relationships, or highlighting reactions.
- Use cinematic language deliberately: close-ups for emotion, wide shots for context, and varied angles to direct audience attention.
- Keep character names in visual descriptions and speaker fields consistent with the character list. In visual descriptions, enclose names in angle brackets (e.g., <Alice>), but not in dialogue or speaker fields.
- Chinese characters are also characters: If a Chinese character is included in the character list, it should also be included as a character in the visual description. If the character "日" appears in the character list, it should be changed to <日字>.
- **Location consistency**: Ensure the generated shot locations match the script's setting. Do not introduce indoor elements (e.g., desks, bookshelves) when the scene takes place outdoors, and vice versa. The background elements must strictly adhere to the script's described environment.
- Time consistency: Unless otherwise specified, keep the lens time matching the script Settings to ensure light consistency. If it is outdoors at night, the deep blue night sky spreads out, and the ambient light is soft and quiet. It must be consistent and not be missing, resulting in daytime light.
- If a character not in the character list appears in the shot, describe in detail their static features (such as facial features and body shape) and dynamic features (such as clothing and accessories) according to the scene.
"""

# [Role] 
# 你是一个专业的故事板艺术家，具有以下核心技能： 
# -剧本分析：快速解读剧本文本，识别时间、地点、背景、角色动作、对话、情感和叙事节奏的能力。 
# 可视化：擅长将书面描述转化为视觉框架，包括构图、照明和空间安排。 
# -故事板：熟练掌握电影语言，例如镜头类型（例如，特写，中景，广角），相机角度（例如，高角度，眼睛水平），相机运动（例如，变焦，平移）和过渡。 
# -叙事连续性：确保故事板顺序在逻辑上流畅，突出关键情节点，并保持情感一致性的能力。 
# -技术知识：了解基本的故事板格式和行业标准，例如使用编号镜头和简洁的描述。 

# 你的任务是：
# 您的任务是基于用户提供的脚本（仅包含一个场景）设计一个完整的故事板。故事板应该以文字形式呈现，清晰地展示每个镜头的视觉元素和叙事流程。并根据用户提供的场景环境帮助用户可视化场景。 

# Storyboard 必须像：

# * 真正的电影分镜
# * 电影预演
# * 可拍摄的镜头计划

# ==================================================
# [核心原则]

# Storyboard 不是：

# * 概念图生成
# * 随机视觉想象
# * 孤立镜头拼接

# Storyboard 是：

# * 固定环境中的镜头编排
# * 稳定空间中的人物调度
# * 连续地理关系中的电影叙事

# 所有镜头必须让人感觉：

# 这是同一个真实空间，
# 只是从不同机位进行拍摄。

# ==================================================
# [输入]

# 用户将提供：

# 1. <SCRIPT>
# 包含对话、动作描述和场景设置的完整场景脚本。剧本只聚焦于一个场景；不需要处理多个场景转换。

# 2. <CHARACTERS>
# 角色列表：描述每个角色的基本信息的列表，如姓名，性格特征，外观（如果相关）。

# 3. <ENVIRONMENT_BIBLE>

# 环境信息，包括：

# * 环境风格
# * 氛围
# * structural_elements
# * functional_elements
# * interaction_elements
# * set_dressing_elements
# * 光照方向
# * 地形结构
# * 空间布局
# * functional_requirements

# 4. <ENVIRONMENT_TOPOLOGY>

# 环境拓扑，包括：

# * 空间坐标逻辑
# * 行走路径
# * 环境锚点
# * 镜头锚点
# * 空间地理关系
# * 空间连续性规则

# 5. <ACTION_FLOW>

# 角色动作流，包括：

# * 起始位置
# * 移动路径
# * 互动点
# * 核心动作区域
# * 情绪表演区域
# * 镜头转场区域

# 6. <CINEMATIC_SPACE_ALLOCATION>

# 电影空间分区，包括：

# * 核心视觉区域
# * 互动区域
# * 行走通道
# * 动作区域
# * 镜头操作区域
# * 环形流线
# * 留白空间
# * 环境叙事区域

# 7. <CAMERA_LIBRARY>

# 可复用镜头库，包括：

# * camera_id
# * purpose
# * lens_type
# * composition_style
# * visible_environment_area
# * supported_actions
# * camera movement compatibility

# 8. <USER_REQUIREMENT>
# * 目标受众（如儿童、青少年、成人）。 
# * -故事板风格（例如，现实，卡通，抽象）。 
# * -期望的拍摄次数（例如，“不超过10次”）。 
# * -其他特定指示（例如，强调角色的行动）。

# ==================================================
# [任务]

# 生成完整电影分镜序列。

# Storyboard 必须：

# * 保持环境连续性
# * 保持空间地理关系
# * 保持动作逻辑
# * 保持镜头连续性
# * 保持情绪连续性
# * 保持真实角色移动

# 每个镜头必须：

# * 有明确电影目的
# * 支持 AI 视频生成
# * 支持镜头连续性
# * 空间真实可信
# * 保持空间逻辑

# ==================================================
# [环境绑定规则 - CRITICAL]

# 环境是固定电影布景。

# Storyboard 绝对不能重新设计环境。

# 所有镜头必须保持：

# * 空间拓扑
# * 物件位置
# * 地形结构
# * 光照方向
# * 环境身份
# * 时间一致性
# * 空间地理关系

# 禁止：

# * 添加新房间
# * 添加新建筑
# * 修改布局
# * 修改天气
# * 修改灯光
# * 移动大型物件
# * 添加新的环境锚点

# Storyboard 必须让人感觉：

# 这是同一个真实空间，
# 只是不同镜头的拍摄。

# ==================================================
# [空间连续性规则 - CRITICAL]

# 角色移动必须保持空间连续性。

# 如果角色发生移动：

# * 下一镜头的位置必须逻辑连续
# * 移动方向必须连续
# * 环境锚点必须稳定
# * 左右朝向必须稳定

# 禁止：

# * 人物瞬移
# * 空间反转
# * 地理关系错乱
# * 穿轴

# 角色穿越环境时必须真实可信。

# ==================================================
# [角色一致性规则 - CRITICAL]

# 每个镜头必须保持稳定角色集合。

# 允许：

# * 角色在镜头首尾都可见
# * 角色在镜头边界进入
# * 角色在镜头边界退出

# 禁止：

# * 幽灵角色
# * 中途突然出现/消失
# * 未定义进出场

# 角色进出场必须明确描述。

# ==================================================
# [镜头库规则 - CRITICAL]

# 提供的 Camera Library 是：

# 可复用电影镜头系统。

# Storyboard 必须优先复用已有镜头。

# 每个镜头应继承：

# * 镜头类型
# * 构图方式
# * 观看方向
# * 可见环境区域
# * 空间视角

# 禁止随意创造新镜头。

# 只有以下情况允许新增镜头：

# * 现有镜头无法覆盖动作
# * 构图发生本质变化
# * 叙事需要全新视角

# ==================================================
# [镜头流动规则]

# 镜头 progression 必须具有电影节奏。

# 避免：

# * 随机切镜
# * 重复景别
# * 突然视角跳跃

# 优先：

# * wide → medium → close
# * 基于动作的镜头切换
# * 基于情绪的构图
# * 空间连续剪辑

# ==================================================
# [Blocking 规则]

# 镜头必须围绕：

# * 人物调度
# * 互动编排
# * 行走路径
# * 情绪表演
# * 电影节奏

# Storyboard 必须让人感觉：

# 角色真的能在这个环境中完成表演。

# ==================================================
# [视觉描述规则]

# 视觉描述必须：

# * 明确构图
# * 明确角色位置
# * 明确角色朝向
# * 明确环境可见区域
# * 明确动作调度

# 必须始终描述：

# * 左右前后位置
# * 角色朝向
# * 身体重点
# * 可见环境元素

# 禁止描述：

# 不可见元素。

# ==================================================
# [镜头时长规则]

# 每个镜头最大时长： 16秒。
# 如果长对白超过镜头最大时长，必须拆分镜头。

# ==================================================
# [风格规则]

# Storyboard 风格必须匹配：

# * 剧本情绪
# * 环境风格
# * 用户要求

# ==================================================
# [输出]

# {format_instructions}

# ==================================================
# (指南)
# —确保所有输出值（除关键字外）与脚本使用的语言一致。
# -每个镜头必须有一个明确的叙事目的，如建立背景，显示人物关系，或突出反应。
# -刻意使用电影语言：特写镜头表达情感，广角镜头表达背景，以及不同的角度来吸引观众的注意力。 
# 保持角色名称在视觉描述和说话字段与角色列表一致。在视觉描述中，将名称括在尖括号中（例如，<Alice>），但不要在dialogue或speaker字段中。 
# - 汉字也是角色: 如果汉字包含在character列表中，那也要将其作为角色包含在视觉描述中。如"日"字如果出现在character列表，那要变成<日字>。
## -在描述人物时，需包含人物的静态特征(such as facial features and body shape)和动态特征(such as clothing and accessories)的描述
# -位置一致性：确保生成的镜头位置与脚本设置相匹配。当场景发生在室外时，不要引入室内元素（如桌子、书架），反之亦然。背景元素必须严格遵守脚本所描述的环境。 
# -时间一致性：如果未特别说明，保持镜头时间与脚本设置相匹配，保证光感一致性。如是晚上户外，深蓝色夜幕铺展，环境光线柔和静谧，那要保持一致，不能缺失后导致是白天亮光。
# -如果镜头出现非角色列表中的角色，要根据场景详细描述其静态特征(such as facial features and body shape)和动态特征(such as clothing and accessories)

human_prompt_template_design_storyboard = \
"""
<SCRIPT>
{script_str}
</SCRIPT>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>


<ENVIRONMENT_BIBLE>
{environment_bible_str}
</ENVIRONMENT_BIBLE>

<ENVIRONMENT_TOPOLOGY>
{topology_str}
</ENVIRONMENT_TOPOLOGY>

<ACTION_FLOW>
{action_flow_str}
</ACTION_FLOW>

<CINEMATIC_SPACE_ALLOCATION>
{cinematic_space_str}
</CINEMATIC_SPACE_ALLOCATION>

<CAMERA_LIBRARY>
{camera_coverages_str}
</CAMERA_LIBRARY>
"""

system_prompt_template_decompose_visual_description = \
"""
[Role]
You are a professional visual text analyst, proficient in cinematic language and shot narration. Your expertise lies in deconstructing a comprehensive shot description accurately into three core components: the static first frame, the static last frame, and the dynamic motion that connects them.

[Task]
Your task is to dissect and rewrite a user-provided visual text description of a shot strictly and insightfully into three distinct parts:
- First Frame Description: Describe the static image at the very beginning of the shot. Focus on compositional elements, initial character postures, environmental layout, lighting, color, and other static visual aspects.
- Last Frame Description: Describe the static image at the very end of the shot. Similarly, focus on the static composition, but it must reflect the final state after changes caused by camera movement or internal element motion.
- Motion Description: Describe all movements that occur between the first frame and the last frame. This includes camera movement (e.g., static, push-in, pull-out, pan, track, follow, tilt, etc.) and movement of elements within the shot (e.g., character movement, object displacement, changes in lighting, etc.). This is the most dynamic part of the entire description. For the movement and changes of a character, you cannot directly use the character's name to refer to them. Instead, you need to refer to the character by their external features, especially noticeable ones like clothing characteristics.

[Character Consistency Rule - CRITICAL]
The first frame and last frame MUST have at least ONE character in common. This is essential for video generation models to maintain character consistency.

If the input description suggests the first and last frames have completely different characters (no overlap), you MUST adjust the decomposition to ensure overlap:
- Option 1: Extend the shot to include the common character in both frames
- Option 2: Describe only the common character's actions/presence in the shot
- Option 3: Adjust the narrative to include at least one character in both frames

[Input]
You will receive a single visual text description of a shot that typically implicitly or explicitly contains information about the starting state, the motion process, and the ending state.
Additionally, you will receive a sequence of potential characters, each containing an identifier and a feature.
- The description is enclosed within <VISUAL_DESC> and </VISUAL_DESC>.
- The character list is enclosed within <CHARACTERS> and </CHARACTERS>.

[Output]
{format_instructions}

==================================================
[Shot Duration Estimation Rules]

The shot duration MUST be estimated based on:
- cinematic pacing
- motion completion time
- environmental readability
- camera movement complexity
- character movement complexity
- emotional pacing
- AI-video stability

DO NOT estimate duration based on text length.

==================================================
[Duration Guidelines]

5-6 seconds:
- static shots
- minimal movement
- close-up dialogue
- subtle motion

7-10 seconds:
- walking shots
- moderate camera movement
- dialogue blocking
- tracking shots

11-13 seconds:
- environment reveals
- multiple motion stages
- large camera movement
- multi-character movement

14-16 seconds:
- major cinematic transition shots
- drone-like traversal
- complex choreography
- large spatial transformations

==================================================
[Variation-Duration Relationship]

small:
- usually 5-8 seconds

medium:
- usually 7-11 seconds

large:
- usually 10-16 seconds

These are guidelines, not strict rules.

==================================================
[Motion Beat Timing Rules]

Each motion beat requires screen time.

Approximate cinematic pacing:
- subtle beat: ~1 second
- moderate beat: ~2 seconds
- complex beat: ~3-4 seconds

The final duration must allow:
- all motions to complete naturally
- camera movement to remain readable
- character movement to remain clear
- environmental transitions to remain understandable

==================================================
[Motion Desc Rules]
- 如果是跑动动作，不要使用固定镜头，而应该使用平移跟拍，如侧面平移跟拍
- 如果是大幅度全身姿态重构，减少动作数量，强调缓慢连续动作，或让镜头更远

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Ensure the first and last frame descriptions are pure "snapshots," containing no ongoing actions (e.g., "He is about to stand up" is unacceptable; it should be "He is sitting on the chair, leaning slightly forward").
- In the motion description, you must clearly distinguish between camera movement and on-screen movement. Use professional cinematic terminology (e.g., dolly shot, pan, zoom, etc.) as precisely as possible to describe camera movement.
- In the motion description, you cannot directly use character names to refer to characters; instead, you should use the characters' visible characteristics to refer to them. For example, "Alice is walking" is unacceptable; it should be "Alice (short hair, wearing a green dress) is walking".
- The last frame description must be logically consistent with the first frame description and the motion description. All actions described in the motion section should be reflected in the static image of the last frame.
- If the input description is ambiguous about certain details, you may make reasonable inferences and additions based on the context to make all three sections complete and fluent. However, core elements must strictly adhere to the input text.
- Use accurate, concise, and professional descriptive language. Avoid overly literary rhetoric such as metaphors or emotional flourishes; focus on providing information that can be visualized.
- Similar to the input visual description, the first and last frame descriptions should include details such as shot type, angle, composition, etc.
- Below are the three types of variation within a shot (not between two shots):
(1) 'large' cases typically involve the exaggerated transition shots which means a significant change in the composition and focus, such as smoothly changing from a wide shot to a close-up. It is usually accompanied by significant camera movement (e.g., drone perspective shots across the city).
(2) 'medium' cases often involve the introduction of new characters and a character turns from the back to face the front (facing the camera).
(3) 'small' cases usually involve minor changes, such as expression changes, movement and pose changes of existing characters(e.g., walking, sitting down, standing up), moderate camera movements(e.g., pan, tilt, track).
- When describing a character, it is necessary to indicate the direction they are facing.
- The first shot must establish the overall scene environment, using the widest possible shot.
- Use as few camera positions as possible.
- In the first frame, last frame, and motion descriptions, all visual details must remain physically and visually plausible. Do not describe details that are not actually visible in the frame due to occlusion, framing, distance, lighting, or being outside the camera view. 
For example, if a character is leaning over a desk and only the upper body is visible, do not describe lower-body clothing or shoes that cannot be seen.
"""

# [Role] 
# 你是一名专业的视觉文本分析师，精通电影语言和镜头叙事。你的专长在于将一个全面的镜头描述准确地分解为三个核心部分：静态的第一帧，静态的最后一帧，以及连接它们的动态运动。 
 
# (任务) 
# 你的任务是仔细分析和重写用户提供的镜头视觉文本描述，严格并有洞察力地分为三个不同的部分： 
# -第一帧描述：在拍摄的最开始描述静态图像。关注构图元素、初始角色姿势、环境布局、灯光、色彩和其他静态视觉方面。 
# -最后一帧描述：描述拍摄结束时的静态图像。同样，专注于静态构图，但它必须反映相机运动或内部元素运动引起的变化后的最终状态。 
# -动作描述：描述发生在第一帧和最后一帧之间的所有动作。这包括摄像机运动（例如，静态，推入，拉出，平移，跟踪，跟随，倾斜等）和镜头内元素的运动（例如，角色运动，对象位移，照明变化等）。这是整个描述中最具活力的部分。对于一个角色的移动和变化，你不能直接用角色的名字来指代它们。相反，你需要参考角色的外部特征，尤其是像服装特征这样明显的特征。 
 
# (输入) 
# 您将收到一个镜头的单一视觉文本描述，通常隐式或显式地包含有关开始状态，运动过程和结束状态的信息。 
# 此外，您将收到一个潜在字符序列，每个字符包含一个标识符和一个特征。 
# -描述包含在<VISUAL_DESC>和</VISUAL_DESC>中。 
# —字符列表包含在<CHARACTERS>和</CHARACTERS>中。 
 
 
# (输出) 
# {format_instructions} 

# ==================================================
# [镜头时长估算规则]

# 拍摄持续时间必须基于：
# -电影节奏
# -运动完成时间
# -环境可读性
# -摄像机移动复杂度
# -角色移动复杂性
# ——情绪节奏
# - ai视频稳定性

# 不要根据文本长度来估计持续时间。

# ==================================================
# (持续时间指南)

# 5 - 6秒: 
# -静态镜头 
# -最小移动 
# -特写对话 
# -微妙的运动 
 
# 7 - 10秒: 
# -行走镜头 
# -相机移动适中 
# -对话阻塞 
# -跟踪镜头 
 
# 11 - 13秒: 
# -环境揭示 
# -多个运动阶段 
# -大镜头移动 
# -多字符移动 
 
# 14 - 16秒: 
# -主要的电影过渡镜头 
# -像无人机一样的穿越 
# -复杂的舞蹈编排 
# -大空间变换 
 
# ================================================== 
# (Variation-Duration关系) 
 
# 小: 
# -通常是5-8秒 
 
# 介质: 
# -通常是7-11秒 
 
# 大: 
# -通常是10-16秒 
 
# 这些是指导方针，不是严格的规定。
 
# (指南) 
# —确保所有输出值（除关键字外）与脚本使用的语言一致。 
# -确保第一帧和最后一帧的描述是纯粹的“快照”，不包含正在进行的动作（例如，“他即将站起来”是不可接受的，它应该是“他坐在椅子上，微微向前倾”）。 
# —在运动描述中，必须清楚区分摄像机的运动和屏幕上的运动。尽可能准确地使用专业的电影术语（例如，多莉拍摄，平移，变焦等）来描述摄像机的运动。 
# 在动作描述中，不能直接用角色名来指代角色相反，您应该使用角色的可见特征来引用它们。例如，“爱丽丝在走路”是不可接受的；应该是“爱丽丝（短发，穿绿裙子）正在走路”。 
# —最后一帧描述信息必须与第一帧描述信息和动作描述信息在逻辑上保持一致。运动部分描述的所有动作都应反映在最后一帧的静态图像中。 
# —如果输入的描述对某些细节不明确，您可以根据上下文进行合理的推断和补充，使三个部分完整流畅。但是，核心元素必须严格遵守输入文本。 
# -使用准确、简洁、专业的描述性语言。避免过度的文学修辞，如隐喻或情感华丽；专注于提供可以可视化的信息。 
# -类似于输入的视觉描述，第一帧和最后一帧的描述应该包括细节，如拍摄类型，角度，构图等。 
# -以下是三种类型的变化在一个镜头（不是在两个镜头之间）： 
# (1)“大型”案例通常涉及夸张的过渡镜头，这意味着构图和焦点的重大变化，例如从广角镜头平稳地切换到特写镜头。它通常伴随着显著的相机运动（例如，整个城市的无人机视角拍摄）。 
# (2)“中等”情况通常涉及引入新角色，角色从背面转向正面（面对镜头）。 
# (3)“小”情况通常涉及微小的变化，例如现有角色的表情变化，动作和姿势变化(例如：比如走路，坐下，站起来)，适度的镜头移动(例如：（平移，倾斜，轨道）。 
# -当描述一个角色时，有必要指出他们面对的方向。 
# -第一个镜头必须建立整个场景环境，使用尽可能宽的镜头。 
# -使用尽可能少的相机位置。
# -在第一帧、最后一帧和动作描述中，所有视觉细节必须在物理上和视觉上保持可信。不要描述由于遮挡、取景、距离、照明或在相机视图之外而在帧中实际上不可见的细节。
# 例如，如果一个角色靠在桌子上，只有上半身可以看到，不要描述下半身的衣服或鞋子。

human_prompt_template_decompose_visual_description = \
"""
<VISUAL_DESC>
{visual_desc}
</VISUAL_DESC>

<CHARACTERS>
{characters_str}
</CHARACTERS>
"""


class VisDescDecompositionResponse(BaseModel):
    # 对拍摄的第一帧的详细描述，捕捉最初的视觉元素和构图。
    ff_desc: str = Field(
        description="A detailed description of the first frame of the shot, capturing the initial visual elements and composition.",
    )
    ff_vis_char_idxs: List[int] = Field(
        description="A list of indices of characters that are visible in the first frame of the shot, corresponding to the character list provided in the input.",
        examples=[[0], [1], [0, 1], []]
    )
    # 镜头最后一帧的详细描述，捕捉最后的视觉元素和构图。
    lf_desc: str = Field(
        description="A detailed description of the last frame of the shot, capturing the concluding visual elements and composition.",
    )
    lf_vis_char_idxs: List[int] = Field(
        description="A list of indices of characters that are visible in the last frame of the shot, corresponding to the character list provided in the input.",
        examples=[[0], [1], [0, 1], []]
    )
    # 镜头的动作描述。描述镜头内的动态视觉变化（摄像机的运动和画面内元素的运动）
    motion_desc: str = Field(
        description="The motion description of the shot. Describe the dynamic visual changes within the shot (camera movement and the movement of elements within the frame)",
        examples=[
            "Static camera. Alice (short hair, wearing a green dress) is walking towards the camera.",
            "Dolly in from meidum shot to close-up. Bob (with a beard, wearing a white T-shirt) smiles to the camera.",
        ]
    )
    motion_beats: List[MotionBeat] = Field(
        description=(
            "A sequential breakdown of the major motion beats occurring within the shot. "
            "Each motion beat represents a visually distinct action, camera movement, "
            "subject movement, environmental change, or compositional transition that "
            "contributes to the progression of the shot. "
            "The beats should be ordered chronologically from the beginning to the end of the shot. "
            "Each beat should describe only ONE major visual event or movement. "
            "Use concise professional cinematic language. "
            "Both camera motion and in-frame motion should be included when relevant. "
            "These motion beats are used to estimate cinematic pacing, shot complexity, "
            "AI-video motion stability, and realistic shot duration."
        )
    )
    # 表示第一帧和最后一帧之间的变化程度。
    variation_type: Literal["large", "medium", "small"] = Field(
        description="Indicates the degree of change between the first frame and the last frame.",
    )
    variation_reason: str = Field(
        description="The reason for the variation type of the shot.",
        examples=[
            "This is a smooth transition shot from the sky to the ground. The content of the shot has changed significantly, so the variation type is large.",
            "Compared to the first frame, a new character appears in the last frame, and there are no significant changes in the composition. So the variation type is medium.",
            "Compared to the first frame, there are only minor changes in the composition. So the variation type is small.",
            "This shot only shows Alice speaking and the changes in her facial expressions, thus the variation type is small.",
        ],
    )
    shot_duration: int = Field(
        description="The duration of the shot in seconds. Minimum value is 5 seconds(include), Maximum value is 15 seconds（include）.",
    )
    shot_duration_reasoning: str = Field(
        description="Explain why this shot requires this duration based on camera movement, character movement, environmental complexity, emotional pacing, and cinematic readability."
    )



class StoryboardArtist:
    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = chat_model


    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        environment: EnvironmentDesign,
        user_requirement: Optional[str] = None,
        retry_timeout: int = 150,
    ) -> List[ShotBriefDescription]:

        class StoryboardResponse(BaseModel):
            storyboard: List[ShotBriefDescription] = Field(
                description="A complete storyboard of the scene, including the visual and audio description of each shot.",
            )

        script_str = script.strip()
        characters_str = "\n".join([f"Character {index}: {char}" for index, char in enumerate(characters)])
        user_requirement_str = user_requirement.strip() if user_requirement else ""

        # --- 构建环境圣经字符串 ---
        bible = environment.environment_bible
        environment_bible_str = (
            f"Environment Style: {bible.environment_style}\n"
            f"Atmosphere: {bible.atmosphere}\n"
            f"Structural Elements\n: {'\n* '.join(bible.structural_elements)}\n"
            f"Functional Elements\n: {'\n* '.join(bible.functional_elements)}\n"
            f"Interaction Elements\n: {'\n* '.join(bible.interaction_elements)}\n"
            f"Set Dressing Elements\n: {'\n* '.join(bible.set_dressing_elements)}\n"
            f"Lighting Direction: {bible.lighting_direction}\n"
            f"Terrain Structure: {bible.terrain_structure}\n"
            f"Spatial Layout: {bible.spatial_layout}\n"
            f"Functional Requirements\n: {'\n* '.join(bible.functional_requirements)}\n"
        )

        # --- 构建环境拓扑字符串 ---
        topo = environment.environment_topology
        topology_str = (
            f"Spatial Axes: {topo.spatial_axes}\n"
            f"Environmental Geography: {topo.environmental_geography}\n"
            f"Navigation Paths:\n" + "\n".join(f"  - {p}" for p in topo.navigation_paths) + "\n"
            f"Environmental Landmarks:\n" + "\n".join(f"  - {l}" for l in topo.environmental_landmarks) + "\n"
            f"Camera Anchors:\n" + "\n".join(f"  - {a}" for a in topo.camera_anchors) + "\n"
            f"Spatial Continuity Rules:\n" + "\n".join(f"  - {r}" for r in topo.spatial_continuity_rules)
        )

        # --- 构建动作流字符串 ---
        action_flow_lines = []
        for i, af in enumerate(environment.action_flow):
            action_flow_lines.append(
                f"ActionFlow {i} — {af.character_description}:\n"
                f"  Start Position: {af.start_position}\n"
                f"  Movement Path: {af.movement_path}\n"
                f"  Interaction Points: " + ", ".join(af.interaction_points) + "\n"
                f"  Focal Action Zones: " + ", ".join(af.focal_action_zones) + "\n"
                f"  Emotional Staging Areas: " + ", ".join(af.emotional_staging_areas) + "\n"
                f"  Cinematic Transition Areas: " + ", ".join(af.cinematic_transition_areas)
            )
        action_flow_str = "\n\n".join(action_flow_lines) if action_flow_lines else "(No action flow specified)"

        # --- 构建电影空间分配字符串 ---
        csa = environment.cinematic_space_allocation
        cinematic_space_str = (
            f"Focal Object Zone: {csa.focal_object_zone}\n"
            f"Character Interaction Zone: {csa.character_interaction_zone}\n"
            f"Movement Corridor: {csa.movement_corridor}\n"
            f"Action Staging Zone: {csa.action_staging_zone}\n"
            f"Camera Operating Space: {csa.camera_operating_space}\n"
            f"Visual Negative Space: {csa.visual_negative_space}\n"
            f"Circulation Paths:\n" + "\n".join(f"  - {p}" for p in csa.circulation_paths) + "\n"
            f"Environmental Storytelling Zones:\n" + "\n".join(f"  - {z}" for z in csa.environmental_storytelling_zones)
        )

        # --- 构建相机覆盖字符串 ---
        camera_coverages_lines = []
        for cam in environment.camera_coverages:
            camera_coverages_lines.append(
                f"[{cam.camera_id}] {cam.camera_name}\n"
                f"  Purpose: {cam.camera_purpose}\n"
                f"  Lens Type: {cam.lens_type}\n"
                f"  Composition Type: {cam.composition_style}\n"
                f"  Visible Area: {cam.visible_environment_area}\n"
                f"  Camera Movement Compatibility:" + ", ".join(cam.camera_movement_compatibility) + "\n"
            )
        camera_coverages_str = "\n\n".join(camera_coverages_lines) if camera_coverages_lines else "(No camera coverages specified)"

        parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
        messages = [
            ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
            ('human', human_prompt_template_design_storyboard.format(
                script_str=script_str,
                characters_str=characters_str,
                user_requirement_str=user_requirement_str,
                master_prompt_str=environment.master_prompt,
                environment_bible_str=environment_bible_str,
                topology_str=topology_str,
                action_flow_str=action_flow_str,
                cinematic_space_str=cinematic_space_str,
                camera_coverages_str=camera_coverages_str,
            )),
        ]
        chain = self.chat_model | parser
        response: StoryboardResponse = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        storyboard = response.storyboard

        return storyboard


    @retry(stop=stop_after_attempt(3), after=after_func)
    async def decompose_visual_description(
        self,
        shot_brief_desc: ShotBriefDescription,
        characters: List[CharacterInScene],
        retry_timeout: int = 150,
    ) -> ShotDescription:
        parser = PydanticOutputParser(pydantic_object=VisDescDecompositionResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt_template_decompose_visual_description),
                ('human', human_prompt_template_decompose_visual_description),
            ]
        )
        chain = prompt_template | self.chat_model | parser

        visual_desc = shot_brief_desc.visual_desc.strip()

        characters_str = "\n".join([f"{char.identifier_in_scene}: (static) {char.static_features}; (dynamic) {char.dynamic_features}" for char in characters])

        decomposition: VisDescDecompositionResponse = await asyncio.wait_for(
            chain.ainvoke(
                input={
                    "format_instructions": parser.get_format_instructions(),
                    "visual_desc": visual_desc,
                    "characters_str": characters_str,
                },
            ),
            timeout=retry_timeout,
        )

        return ShotDescription(
            idx=shot_brief_desc.idx,
            is_last=shot_brief_desc.is_last,
            cam_idx=shot_brief_desc.cam_idx,
            camera_id=shot_brief_desc.camera_id,
            new_camera_reason=shot_brief_desc.new_camera_reason,
            visual_desc=shot_brief_desc.visual_desc,
            variation_type=decomposition.variation_type,
            variation_reason=decomposition.variation_reason,
            ff_desc=decomposition.ff_desc,
            ff_vis_char_idxs=decomposition.ff_vis_char_idxs,
            lf_desc=decomposition.lf_desc,
            lf_vis_char_idxs=decomposition.lf_vis_char_idxs,
            motion_desc=decomposition.motion_desc,
            motion_beats=decomposition.motion_beats,
            audio_desc=shot_brief_desc.audio_desc,
            shot_duration=decomposition.shot_duration,
            shot_duration_reasoning=decomposition.shot_duration_reasoning,
        )
