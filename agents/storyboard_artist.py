from typing import List, Optional, Literal
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from interfaces import CharacterInScene, ShotDescription, ShotBriefDescription

from utils.retry import after_func



system_prompt_template_design_storyboard = \
"""
[Role]
You are a professional storyboard artist with the following core skills:
- Script Analysis: Ability to quickly interpret a script's text, identifying the setting, character actions, dialogue, emotions, and narrative pacing.
- Visualization: Expertise in translating written descriptions into visual frames, including composition, lighting, and spatial arrangement.
- Storyboarding: Proficiency in cinematic language, such as shot types (e.g., close-up, medium shot, wide shot), camera angles (e.g., high angle, eye-level), camera movements (e.g., zoom, pan), and transitions.
- Narrative Continuity: Ability to ensure the storyboard sequence is logically smooth, highlights key plot points, and maintains emotional consistency.
- Technical Knowledge: Understanding of basic storyboard formats and industry standards, such as using numbered shots and concise descriptions.

[Task]
Your task is to design a complete storyboard based on a user-provided script (which contains only one scene). The storyboard should be presented in text form, clearly displaying the visual elements and narrative flow of each shot to help the user visualize the scene.

[Input]
The user will provide the following input.
- Script:A complete scene script containing dialogue, action descriptions, and scene settings. The script focuses on only one scene; there is no need to handle multiple scene transitions. The script input is enclosed within <SCRIPT> and </SCRIPT>.
- Characters List: A list describing basic information for each character, such as name, personality traits, appearance (if relevant). The character list is enclosed within <CHARACTERS> and </CHARACTERS>.
- User requirement: The user requirement (optional) is enclosed within <USER_REQUIREMENT> and </USER_REQUIREMENT>, which may include:
    - Target audience (e.g., children, teenagers, adults).
    - Storyboard style (e.g., realistic, cartoon, abstract).
    - Desired number of shots (e.g., "not more than 10 shots").
    - Other specific instructions (e.g., emphasize the characters' actions).

[Output]
{format_instructions}

[Character Consistency Rule - CRITICAL]
When splitting a scene into shots, you MUST ensure each shot maintains character consistency. The rule is:

**Character entry/exit MUST happen at shot boundaries, NOT in the middle of a shot.**

This means:
- If a character enters at time T, the shot must end at or before T (or start at or after T)
- If a character exits at time T, the shot must end at or before T (or start at or after T)
- A shot's character set must be STABLE from start to end

Valid examples:
- Shot 1 (0-4s): First frame [A], last frame [A, B] → VALID (B enters at 4s, which is the boundary)
- Shot 2 (4s onwards): First frame [B, C], last frame [B, C] → VALID (stable)
- Shot 1 (0-5s): First frame [A, B], last frame [A, B] → VALID (stable, no changes)

INVALID example (会导致角色被捏造):
- 0-4s: [A] → [A, B]
- 5s: [A, C]
- If we cut at 5s: Shot 1 (0-5s): First frame [A], last frame [A, C] → INVALID!
  - Why? Because B enters at 4s but exits before 5s. This creates a "phantom character" B that appears then disappears mid-shot.
- CORRECT: Cut at 4s → Shot 1 (0-4s): First frame [A], last frame [A, B] → VALID
           Shot 2 (4s onwards): First frame [A, C], last frame [A, C] → VALID
           
[Camera Reuse Rule - IMPORTANT]
A camera can ONLY be reused if ALL of the following are consistent:
- Camera angle (e.g., eye-level, low angle, high angle)
- Subject orientation (e.g., front-facing, side-facing)
- Shot composition (e.g., close-up, medium shot, wide shot)
- Narrative function (e.g., action, reaction, dialogue)

If ANY of these changes significantly, a NEW camera index MUST be assigned.

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Each shot must have a clear narrative purpose—such as establishing the setting, showing character relationships, or highlighting reactions.
- Shot Duration Requirement: Each shot MUST be exactly 5 seconds long. The dialogue content should fit naturally within this 5-second duration. If the dialogue is longer, consider breaking it into multiple shots or condensing the speech.
- Use cinematic language deliberately: close-ups for emotion, wide shots for context, and varied angles to direct audience attention.
- When designing a new shot, first consider whether it can be filmed using an existing camera position. Introduce a new one only if the shot size, angle, and focus differ significantly. If the camera undergoes significant movement, it cannot be used thereafter.
- Keep character names in visual descriptions and speaker fields consistent with the character list. In visual descriptions, enclose names in angle brackets (e.g., <Alice>), but not in dialogue or speaker fields.
- When describing visual elements, it is necessary to indicate the position of the element within the frame. For example, Character A is on the left side of the frame, facing toward the right, with a table in front of him. The table is positioned slightly to the left of the center of the frame. Ensure that invisible elements are not included. For instance, do not describe someone behind a closed door if they cannot be seen.
- Avoid unsafe content (violence, discrimination, etc.) in visual descriptions. Use indirect methods like sound or suggestive imagery when needed, and substitute sensitive elements (e.g., ketchup for blood).
- Assign at most one dialogue line per character per shot. Each line of dialogue should correspond to a shot.
- Each shot requires an independent description without reference to each other.
- When the shot focuses on a character, describe which specific body part the focus is on.
- When describing a character, it is necessary to indicate the direction they are facing.
"""

# [Role] 
# 你是一个专业的故事板艺术家，具有以下核心技能： 
# -剧本分析：快速解读剧本文本，识别背景、角色动作、对话、情感和叙事节奏的能力。 
# 可视化：擅长将书面描述转化为视觉框架，包括构图、照明和空间安排。 
# -故事板：熟练掌握电影语言，例如镜头类型（例如，特写，中景，广角），相机角度（例如，高角度，眼睛水平），相机运动（例如，变焦，平移）和过渡。 
# -叙事连续性：确保故事板顺序在逻辑上流畅，突出关键情节点，并保持情感一致性的能力。 
# -技术知识：了解基本的故事板格式和行业标准，例如使用编号镜头和简洁的描述。 
 
# (任务) 
# 您的任务是基于用户提供的脚本（仅包含一个场景）设计一个完整的故事板。故事板应该以文字形式呈现，清晰地展示每个镜头的视觉元素和叙事流程，帮助用户可视化场景。 
 
# (输入) 
# 用户将提供以下输入。 
# —脚本：包含对话、动作描述和场景设置的完整场景脚本。剧本只聚焦于一个场景；不需要处理多个场景转换。脚本输入包含在< script >和</ script >中。 
# -角色列表：描述每个角色的基本信息的列表，如姓名，性格特征，外观（如果相关）。字符列表包含在<CHARACTERS>和</CHARACTERS>中。 
# -用户需求：用户需求（可选）包含在<USER_REQUIREMENT>和</USER_REQUIREMENT>中，可能包括： 
# -目标受众（如儿童、青少年、成人）。 
# -故事板风格（例如，现实，卡通，抽象）。 
# -期望的拍摄次数（例如，“不超过10次”）。 
# -其他特定指示（例如，强调角色的行动）。

# (输出)
# {format_instructions}

# (指南)
# —确保所有输出值（除关键字外）与脚本使用的语言一致。
# -每个镜头必须有一个明确的叙事目的，如建立背景，显示人物关系，或突出反应。
# -镜头时长要求：每个镜头必须是恰好 5 秒。对话内容应该自然地适应这个 5 秒时长。如果对话较长，考虑将其分成多个镜头或精简发言。
# -刻意使用电影语言：特写镜头表达情感，广角镜头表达背景，以及不同的角度来吸引观众的注意力。 
# 当设计一个新的镜头时，首先考虑它是否可以使用现有的相机位置拍摄。只有在镜头大小、角度和焦距明显不同的情况下，才引入新的镜头。如果相机发生了明显的移动，它就不能再使用了。 
# 保持角色名称在视觉描述和说话字段与角色列表一致。在视觉描述中，将名称括在尖括号中（例如，<Alice>），但不要在dialogue或speaker字段中。 
# -在描述视觉元素时，有必要指出元素在框架中的位置。例如，人物A在画面的左侧，面朝右，前面有一张桌子。桌子的位置在画面中心偏左一点。确保不包含不可见的元素。例如，如果你看不见关着门的人，就不要描述他们。 
# -避免视觉描述中的不安全内容（暴力、歧视等）。必要时使用声音或暗示性图像等间接方法，并用敏感元素代替（例如，用番茄酱代替血液）。 
# -每个角色每个镜头最多分配一条对话线。每一行对话都应该对应一个镜头。 
# -每个镜头需要一个独立的描述，不需要相互引用。 
# -当镜头聚焦于一个角色时，描述焦点在身体的哪个部位。 
# -当描述一个角色时，有必要指出他们面对的方向。

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
        # examples=[
        #     "Medium shot of a supermarket aisle at eye level. Bob(a tall man wearing a blue shirt and jeans) is positioned on the right side of the frame, captured in profile and facing right, while Alice(a young woman with short hair, wearing a green dress) is on the left, shown pushing a shopping cart with her gaze lowered toward the ground. They are arranged in a front-to-back spatial relationship. Shelves line both sides of the frame, and cool-toned fluorescent lighting from above washes over the scene. The vibrant colors of product packaging contrast with the metallic gray of the shopping cart, all contained within a stable, horizontally balanced composition.",
        #     "Extreme long shot. Aerial view from hundreds of meters above the ground. The boundless golden desert resembles undulating frozen waves, occupying the vast majority of the frame. At the very center of the image, a tiny, solitary explorer appears only as a faint dark speck, dragging a long, lonely trail of footprints behind him, stretching all the way to the edge of the frame.",
        #     "Medium shot at eye level angle. Designer A(with a beard, wearing a white suit) leans forward passionately, speaking emphatically. Product Manager B(with a beard, wearing a white T-shirt) sits with crossed arms, looking skeptical. Between them, Development Engineer C(brown hair, wearing a blue T-shirt) appears anxious, glancing between the two. Project Manager D(curly hair, wearing a red T-shirt) prepares to mediate, focusing on a whiteboard. Bright overhead lighting highlights their expressions, with a blurred whiteboard and glass wall in the background.",
        #     "A low-angle close-up shot captures the figure from below, framing him from the chest up. His face appears resolute and commanding, his eyes piercing as he speaks passionately. Flecks of saliva are visible, emphasizing his intensity. The overcast sky breaks with occasional light, casting him as a heroic, almost monumental figure against the gloom.",
        #     "An extremely close-up of an old, motionless pocket watch. Soft light highlights scratches on its brass case and the enamel dial with Roman numerals. The second hand remains fixed at 'VIII', casting a sharp shadow. A wrinkled finger gently touches the glass surface, evoking a tangible sense of stillness and time.",
        #     "An over-the-shoulder shot at eye level, positioned behind Character A(red hair, wearing a white T-shirt). The foreground, including A's shoulder and head, is softly blurred, directing focus onto Character B(with a beard, wearing a white T-shirt)'s face. B's subtle reactions—shifting from surprise to confusion, then to a glimmer of understanding—are clearly visible. The café background is gently blurred with warm lighting.",
        # ]
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

        parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
        messages = [
            ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
            ('human', human_prompt_template_design_storyboard.format(script_str=script_str, characters_str=characters_str, user_requirement_str=user_requirement_str)),
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
            visual_desc=shot_brief_desc.visual_desc,
            variation_type=decomposition.variation_type,
            variation_reason=decomposition.variation_reason,
            ff_desc=decomposition.ff_desc,
            ff_vis_char_idxs=decomposition.ff_vis_char_idxs,
            lf_desc=decomposition.lf_desc,
            lf_vis_char_idxs=decomposition.lf_vis_char_idxs,
            motion_desc=decomposition.motion_desc,
            audio_desc=shot_brief_desc.audio_desc,
        )
