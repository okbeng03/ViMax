import logging
from typing import List, Tuple
from tenacity import retry, stop_after_attempt
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from utils.image import image_path_to_b64

from utils.retry import after_func
from utils.provider_presets import create_chat_model
from utils.completion_logger import log_agent

system_prompt_template_select_reference_images_only_text = \
"""
[Role]
You are a professional visual creation assistant skilled in multimodal image analysis and reasoning.

[Task]
Your core task is to intelligently select the most suitable reference images from a provided set of reference image descriptions (including multiple character reference images、Chinese characters reference images、environmental reference image and existing scene images from prior frames) based on the user's text description (describing the target frame), ensuring that the subsequently generated image meets the following key consistencies:
- Character Consistency: The appearance (e.g. gender, ethnicity, age, facial features, hairstyle, body shape), clothing, expression, posture, etc., of the generated character should highly match the reference image descriptions.
- Chinese characters Consistency: The generated Chinese characters should be consistent with the image of the shape of the character in the corresponding period.
- Environmental Consistency: The scene of the generated image (e.g., background, lighting, atmosphere, layout) should remain coherent with the existing image descriptions from prior frames.
- preserve spatial continuity
- Style Consistency: The visual style of the generated image (e.g., realistic, cartoon, film-like, color tone) should harmonize with the reference image descriptions.

[Input]
You will receive a text description of the target frame, along with a sequence of reference image descriptions.
- The text description of the target frame is enclosed within <FRAME_DESC> and </FRAME_DESC>.
- The sequence of reference image descriptions is enclosed within <SEQ_DESC> and </SEQ_DESC>. Each description is prefixed with its index, starting from 0.

Below is an example of the input format:
<FRAME_DESC>
[Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob's expression shifts from surprise to delight as he recognizes Alice.
</FRAME_DESC>

<SEQ_DESC>
Image 1: A front-view portrait of Alice.
Image 2: A front-view portrait of Bob.
Image 3: 甲骨文 of the character 日.如果未明确，泛指古人的汉字
Image 4: 楷书、简体 of the character 日.如果未明确，泛指现在的汉字
Image 5: [Camera 0] Medium shot of the supermarket aisle. Alice and Bob are shown in profile facing the right side of the frame. Bob is on the right side of the frame, and Alice is on the left side. Alice, looking down and pushing a shopping cart, follows closely behind Bob and accidentally bumps into his heel.
Image 6: [Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob quickly turns around, and his expression shifts from neutral to surprised.
Image 7: [Camera 2] Shot from Bob's over-the-shoulder perspective. Bob is on the side closer to the camera, with only his shoulder appearing in the lower right corner of the frame. Alice is on the side farther from the camera, positioned slightly left of center in the frame. Alice looks down, then up as she prepares to apologize. Upon realizing it's someone familiar, her expression shifts to one of surprise.
</SEQ_DESC>


[Output]
You need to select up to 8 of the most relevant reference images based on the user's description and put the corresponding indices in the ref_image_indices field of the output. At the same time, you should generate a text prompt that describes the image to be created, specifying which elements in the generated image should reference which image description (and which elements within it).

{format_instructions}

[Text Prompt RULES]
The text_prompt MUST begin with a global consistency instruction.

Example:

"Maintain strict consistency with the provided reference images, including character identity, costume design, environment layout, lighting direction, cinematic style, camera continuity, and spatial continuity. Preserve the same cinematic world and visual atmosphere."

After the consistency instruction, describe ONLY the CURRENT FRAME STATE.

The text_prompt should primarily describe:

- shot type
- framing
- composition
- visible character positions
- character orientation
- body direction
- body balance and weight
- visible posture
- hand positions
- visible body regions
- visible facial expressions
- eye direction
- current action moment
- interaction relationships
- foreground/background spatial layering
- cinematic blocking
- composition relationships

Only minimally reference environment elements name required for:
- grounding characters spatially
- supporting interactions
- anchoring composition

Don't reference the element detail.
The environment itself must NOT be re-described.

text_prompt 最后以参考图片引用描述结束，引用的 index 基于 ref_image_indices（而不是 SEQ_DESC indices）：

如：
"小豆丁人物特征、穿着参考 Image 0。\n字博士人物特征、穿着参考 Image 1。\n整体环境、背景参考 Image 2。"

[visual_desc RULES]

The visual_desc must:

- be objective
- be visually precise
- obey physical visibility logic
- obey occlusion logic
- preserve spatial continuity
- preserve camera continuity

Focus primarily on:

- visible character states
- visible poses
- visible body regions
- visible facial expressions
- visible hand states
- visible interaction states
- framing
- composition
- foreground/background layering
- depth relationships
- spatial positioning
- cinematic blocking

Only describe environment elements
that are physically visible
and compositionally relevant.

Do NOT:
- redesign the environment
- infer unseen areas
- describe hidden objects
- restate full environment layouts

[Guidelines]
- Ensure that the language of all output values (not include keys) matches that used in the frame description.
- The reference image descriptions may depict the same character from different angles, in different outfits, or in different scenes. Identify the description closest to the version described by the user
- Prioritize image descriptions with similar compositions, i.e., shots taken by the same camera.
- The images from prior frames are arranged in chronological order. Give higher priority to more recent images (those closer to the end of the sequence).
- Choose reference image descriptions that are as concise as possible and avoid including duplicate information. For example, if Image 3 depicts the facial features of Bob from the front, and Image 1 also depicts Bob's facial features from the front-view portrait, then Image 1 is redundant and should not be selected.
- When a new character appears in the frame description, prioritize selecting their portrait image description (if available) to ensure accurate depiction of their appearance. Pay attention to whether the character is facing the camera from the front, side, or back. Choose the most suitable view as the reference image for the character.
- For character portraits, you can only select at most one image from multiple views (front, side, back). Choose the most appropriate one based on the frame description. For example, when depicting a character from the side, choose the side view of the character.
- Note: The focus of the lens should be on the upper or lower body area. Do not abruptly present a truncated body, which is half of the meaning of the lens. Maintain a three-dimensional proportion. Avoid presenting a body that is cut off at the waist or legs.
- Select at most **8** optimal reference image descriptions.
- **没有出现在参考图里的角色，必须保持详细描述其静态&动态特征**
"""

# [Role] 
# 你是一个专业的视觉创作助理，擅长多模态图像分析和推理。 
 
# (任务) 
# 您的核心任务是根据用户的文本描述（描述目标帧），从提供的一组参考图像描述（包括多个字符参考图像、场景环境图像和先前帧的现有场景图像）中智能地选择最合适的参考图像，确保随后生成的图像满足以下关键一致性： 
# -角色一致性：生成角色的外观（如性别，种族，年龄，面部特征，发型，体型），服装，表情，姿势等应与参考图像描述高度匹配。 
# -汉字一致性: 生成的汉字要与该汉字对应时期的字形的图像保持一致。
# -环境一致性：生成图像的场景（例如，背景，照明，氛围，布局）应与先前帧的现有图像描述保持一致。 
# - 保持镜头连续性
# -风格一致性：生成图像的视觉风格（例如，逼真的，卡通的，电影般的，色调）应该与参考图像描述协调一致。 
 
# (输入) 
# 您将收到目标帧的文本描述，以及一系列参考图像描述。 
# -目标帧的文本描述包含在<FRAME_DESC>和</FRAME_DESC>中。 
# -参考图像描述序列包含在<SEQ_DESC>和</SEQ_DESC>中。每个描述都有其索引前缀，从0开始。 
 
# 下面是输入格式的示例： 
# < FRAME_DESC > 
# [镜头1]从爱丽丝肩后视角拍摄。爱丽丝在靠近相机的一侧，只有她的肩膀出现在画面的左下角。鲍勃站在离相机较远的那一边，在画面中心稍右的位置。当鲍勃认出爱丽丝时，他的表情从惊讶变成了高兴。 
# < / FRAME_DESC > 
 
# < SEQ_DESC > 
# 图0：爱丽丝的正面肖像。 
# 图1：鲍勃的正面肖像。 
# 图2:[摄像头0]超市过道的中景。爱丽丝和鲍勃的侧面面向画面的右侧。鲍勃在画框的右边，爱丽丝在左边。爱丽丝低头推着购物车，紧紧跟在鲍勃后面，不小心撞到了他的脚跟。 
# 图3:[相机1]从爱丽丝的肩膀上拍摄的。爱丽丝在靠近相机的一侧，只有她的肩膀出现在画面的左下角。鲍勃站在离相机较远的那一边，在画面中心稍右的位置。鲍勃迅速转过身来，他的表情从中性变成了惊讶。 
# 图4:[相机2]从鲍勃的肩膀上拍摄的。鲍勃在靠近相机的那一边，只有他的肩膀出现在画面的右下角。爱丽丝站在离相机较远的那一边，位于画面中心略偏左的位置。爱丽丝低下头，然后抬起头，准备道歉。当意识到是熟人时，她的表情变成了惊讶。 
# < / SEQ_DESC > 
 
 
# (输出) 
# 您需要根据用户的描述选择最多8个最相关的参考图像，并将相应的索引放在输出的ref_image_indices字段中。同时，您应该生成一个描述要创建的图像的文本提示符，指定生成的图像中的哪些元素应该引用哪个图像描述（以及其中的哪些元素）。 
 
# {format_instructions} 
 
# (指南) 
# —确保所有输出值（不包括键）的语言与帧描述中使用的语言匹配。 
# -参考图像描述可以从不同角度，不同服装或不同场景描绘同一角色。确定与用户描述的版本最接近的描述 
# -优先考虑具有相似构图的图像描述，即由同一相机拍摄的照片。 
# -先前帧的图像按时间顺序排列。给最近的图像更高的优先级（那些更接近序列的末尾）。 
# —参考图片描述尽量简洁，避免重复信息。例如，如果图像3从正面描绘了Bob的面部特征，而图像1也从正面描绘了Bob的面部特征，那么图像1是冗余的，不应该被选中。 
# 当一个新角色出现在帧描述中，优先选择他们的肖像图像描述（如果可用），以确保准确描述他们的外表。注意角色是正面、侧面还是背面面对镜头。选择最合适的视图作为角色的参考图像。 
# -对于人物肖像，您最多只能从多个视图（正面，侧面，背面）中选择一个图像。根据框架描述选择最合适的一个。例如，当从侧面描绘一个角色时，选择角色的侧面视图。 
# —最多选择“**8** opt”

system_prompt_template_select_reference_images_multimodal = \
"""
[Role]
You are a professional visual creation assistant skilled in multimodal image analysis and reasoning.

[Task]
Your core task is to intelligently select the most suitable reference images from a provided reference image library (including multiple character reference images and existing scene images from prior frames) based on the user's text description (describing the target frame), ensuring that the subsequently generated image meets the following key consistencies:
- Character Consistency: The appearance (e.g. gender, ethnicity, age, facial features, hairstyle, body shape), clothing, expression, posture, etc., of the generated character should highly match the reference images.
- Environmental Consistency: The scene of the generated image (e.g., background, lighting, atmosphere, layout) should remain coherent with the existing images from prior frames.
- Style Consistency: The visual style of the generated image (e.g., realistic, cartoon, film-like, color tone) should harmonize with the reference images and existing images.

[Input]
You will receive a text description of the target frame, along with a sequence of reference images.
- The text description of the target frame is enclosed within <FRAME_DESC> and </FRAME_DESC>.
- The sequence of reference images is enclosed within <SEQ_IMAGES> and </SEQ_IMAGES>. Each reference image is provided with a text description. The reference images are indexed starting from 0.

Below is an example of the input format:
<FRAME_DESC>
[Camera 1] Shot from Alice's over-the-shoulder perspective. <Alice> is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. <Bob> is on the side farther from the camera, positioned slightly right of center in the frame. <Bob>'s expression shifts from surprise to delight as he recognizes <Alice>.
</FRAME_DESC>

<SEQ_IMAGES>
Image 0: A front-view portrait of Alice.
[Image 0 here]
Image 1: A front-view portrait of Bob.
[Image 1 here]
Image 2: [Camera 0] Medium shot of the supermarket aisle. Alice and Bob are shown in profile facing the right side of the frame. Bob is on the right side of the frame, and Alice is on the left side. Alice, looking down and pushing a shopping cart, follows closely behind Bob and accidentally bumps into his heel.
[Image 2 here]
Image 3: [Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob is back to the camera.
[Image 3 here]
Image 4: [Camera 2] Shot from Bob's over-the-shoulder perspective. Bob is on the side closer to the camera, with only his shoulder appearing in the lower right corner of the frame. Alice is on the side farther from the camera, positioned slightly left of center in the frame. Alice looks down, then up as she prepares to apologize. Upon realizing it's someone familiar, her expression shifts to one of surprise.
</SEQ_IMAGES>

[Output]
You need to select the most relevant reference images based on the user's description and put the corresponding indices in the `ref_image_indices` field of the output. At the same time, you should generate a text prompt that describes the image to be created, specifying which elements in the generated image should reference which image (and which elements within it).

{format_instructions}

[Guidelines]
- Ensure that the language of all output values (not include keys) matches that used in the frame description.
- The reference image descriptions may depict the same character from different angles, in different outfits, or in different scenes. Identify the description closest to the version described by the user
- Prioritize image descriptions with similar compositions, i.e., shots taken by the same camera.
- The images from prior frames are arranged in chronological order. Give higher priority to more recent images (those closer to the end of the sequence).
- Choose reference image descriptions that are as concise as possible and avoid including duplicate information. For example, if Image 3 depicts the facial features of Bob from the front, and Image 1 also depicts Bob's facial features from the front-view portrait, then Image 1 is redundant and should not be selected.
- For character portraits, you can only select at most one image from multiple views (front, side, back). Choose the most appropriate one based on the frame description. For example, when depicting a character from the side, choose the side view of the character.
- Select at most **8** optimal reference image descriptions.
- The text guiding image editing should be as concise as possible.
"""

# [Role] 
# 你是一个专业的视觉创作助理，擅长多模态图像分析和推理。 
 
# (任务) 
# 您的核心任务是根据用户的文本描述（描述目标帧），从提供的参考图片库（包括多个字符参考图像和先前帧的现有场景图像）中智能地选择最合适的参考图像，确保随后生成的图像满足以下关键一致性： 
# -角色一致性：生成角色的外观（如性别，种族，年龄，面部特征，发型，体型），服装，表情，姿势等应与参考图像高度匹配。 
# -环境一致性：生成图像的场景（例如，背景，照明，氛围，布局）应该与先前帧的现有图像保持一致。 
# -风格一致性：生成图像的视觉风格（如写实、卡通、电影式、色调）应与参考图像和现有图像协调一致。 
 
# (输入) 
# 您将收到目标帧的文本描述，以及一系列参考图像。 
# -目标帧的文本描述包含在<FRAME_DESC>和</FRAME_DESC>中。 
# —参考图像序列包含在<SEQ_IMAGES>和</SEQ_IMAGES>中。每个参考图像都提供了文本描述。参考图像从0开始索引。 
 
# 下面是输入格式的示例： 
# < FRAME_DESC > 
# [镜头1]从爱丽丝肩后视角拍摄。Alice b>在离相机更近的一侧，只有她的肩膀出现在画面的左下角。<Bob>在离相机较远的一侧，位于画面中心稍右的位置。鲍勃的表情从惊讶变成了喜悦，因为他认出了爱丽丝。 
# < / FRAME_DESC > 
 
# < SEQ_IMAGES > 
# 图0：爱丽丝的正面肖像。 
# [图0] 
# 图1：鲍勃的正面肖像。 
# [图1] 
# 图2:[摄像头0]超市过道的中景。爱丽丝和鲍勃的侧面面向画面的右侧。鲍勃在画框的右边，爱丽丝在左边。爱丽丝低头推着购物车，紧紧跟在鲍勃后面，不小心撞到了他的脚跟。 
# [图2] 
# 图3:[相机1]从爱丽丝的肩膀上拍摄的。爱丽丝在靠近相机的一侧，只有她的肩膀出现在画面的左下角。鲍勃站在离相机较远的那一边，在画面中心稍右的位置。鲍勃回到镜头前。 
# [图3] 
# 图4:[相机2]从鲍勃的肩膀上拍摄的。鲍勃在靠近相机的那一边，只有他的肩膀出现在画面的右下角。爱丽丝站在离相机较远的那一边，位于画面中心略偏左的位置。爱丽丝低下头，然后抬起头，准备道歉。当意识到是熟人时，她的表情变成了惊讶。 
# < / SEQ_IMAGES > 
 
# (输出) 
# 您需要根据用户的描述选择最相关的参考图像，并将相应的索引放在输出的‘ ref_image_indices ’字段中。同时，您应该生成一个描述要创建的图像的文本提示符，指定生成的图像中的哪些元素应该引用哪个图像（以及其中的哪些元素）。 
 
# {format_instructions} 
 
 
# (指南) 
# —确保所有输出值（不包括键）的语言与帧描述中使用的语言匹配。 
# -参考图像描述可以从不同角度，不同服装或不同场景描绘同一角色。确定与用户描述的版本最接近的描述 
# -优先考虑具有相似构图的图像描述，即由同一相机拍摄的照片。 
# -先前帧的图像按时间顺序排列。给最近的图像更高的优先级（那些更接近序列的末尾）。 
# —参考图片描述尽量简洁，避免重复信息。例如，如果图像3从正面描绘了Bob的面部特征，而图像1也从正面描绘了Bob的面部特征，那么图像1是冗余的，不应该被选中。 
# -对于人物肖像，您最多只能从多个视图（正面，侧面，背面）中选择一个图像。根据框架描述选择最合适的一个。例如，当从侧面描绘一个角色时，选择角色的侧面视图。 
# -最多选择**8**最佳参考图像描述。 
# -指导图像编辑的文本应尽可能简洁。

human_prompt_template_select_reference_images = \
"""
<FRAME_DESC>
{frame_description}
</FRAME_DESC>

<SEQ_IMAGES>
{image_descriptions}
<SEQ_IMAGES>
"""

class RefImageIndicesAndTextPrompt(BaseModel):
    """
    Reference-image selection result and cinematic frame-state prompts.
    """

    ref_image_indices: List[int] = Field(
        description=(
            "Indices of the selected reference images from the provided sequence. **The index must starts from 0.**"
        ),
        examples=[
            [0, 2, 5]
        ]
    )

    text_prompt: str = Field(
        description=(
            "A cinematic frame-state prompt used for AI image generation.You need to describe the image to be generated, specifying which elements in the generated image should reference which image (and which elements within it).这里，参考映像的索引应该指向它在ref_image_indices列表中的位置，而不是提供的映像列表中的序列号。参考图片必须是图片n的格式，不要使用除图片以外的任何其他词。不存在的人物不要描述参考引用。"
        ),
        examples=[
            (
                "保持与提供的参考图严格一致，包括角色身份、服装设计、环境布局、光照方向、电影风格、镜头连续性与空间连续性。"
                "保持同一个电影世界与视觉氛围。\n\n"
                "中景镜头，三分法则构图。"
                "小豆丁坐在画面左侧的木制书法桌前。"
                "他的上半身略微前倾。"
                "右手握着毛笔停顿在宣纸上方。"
                "眉头紧锁，视线落在纸上的复杂汉字上。"
                "桌面散落着练字纸张与墨迹。"
                "背景中的实验室墙壁与汉字演变挂图轻微虚化。"
                "左上方窗户投射出暖色体积光。\n"
                "男孩的外貌特征和穿着参考 Image 0。环境布局、背景整体参考 Image 1。"
            )
        ]
    )

    visual_desc: str = Field(
        description=(
            "A complete visual description used for image quality analysis"
        ),
        examples=[
            (
                "中景镜头，三分法则构图。"
                "小豆丁位于画面左侧，坐在木制书法桌前。"
                "画面中仅可见他的上半身。"
                "他身体略微前倾，右手握着毛笔悬停在宣纸上方。"
                "眉头紧锁，嘴巴微微撅起，视线集中在桌面的复杂汉字上。"
                "木桌位于左前区域，桌面摆放着砚台、毛笔与散落的宣纸。"
                "实验室背景轻微虚化，后方墙壁挂着汉字演变挂图。"
                "左上方高处窗户投射出暖色体积光，空气中可见轻微灰尘颗粒。"
            )
        ]
    )


class ReferenceImageSelector:
    def __init__(
        self,
        chat_model,
    ):
        self.chat_model = create_chat_model(
            model_provider="qwen",
            model="deepseek-v4-flash",
        )

    @log_agent("ReferenceImageSelector")
    @retry(
        stop=stop_after_attempt(1),
        after=after_func,
    )
    async def select_reference_images_and_generate_prompt(
        self,
        available_image_path_and_text_pairs: List[Tuple[str, str]],
        frame_description: str,
        only_text_model: bool = False,
    ):
        filtered_image_path_and_text_pairs = available_image_path_and_text_pairs

        # 1. filter images using text-only model
        # 如果参考图片数量大于等于8张，则使用text-only model进行过滤，得到8张图片
        if len(available_image_path_and_text_pairs) >= 8 or only_text_model:
            human_content = []
            for idx, (_, text) in enumerate(available_image_path_and_text_pairs):
                human_content.append(f"Image {idx}: {text}")
            parser = PydanticOutputParser(pydantic_object=RefImageIndicesAndTextPrompt)

            messages = [
                SystemMessage(content=system_prompt_template_select_reference_images_only_text.format(format_instructions=parser.get_format_instructions())),
                HumanMessage(content=human_prompt_template_select_reference_images.format(frame_description=frame_description, image_descriptions="\n".join(human_content)))
            ]

            chain = self.chat_model | parser

            try:
                ref = await chain.ainvoke(messages)
                filtered_image_path_and_text_pairs = [available_image_path_and_text_pairs[i] for i in ref.ref_image_indices]
                logging.info(f"Filtered image idx:{ref.ref_image_indices}:{filtered_image_path_and_text_pairs}")
                
                if only_text_model:
                    return {
                        "reference_image_path_and_text_pairs": filtered_image_path_and_text_pairs,
                        "text_prompt": ref.text_prompt,
                    }
                
            except Exception as e:
                logging.error(f"Error get image prompt: \n{e}")
                raise e
        
        # 2. filter images using multimodal model
        human_content = []
        for idx, (image_path, text) in enumerate(filtered_image_path_and_text_pairs):
            human_content.append(f"Image {idx}: {text}")
            human_content.append(image_path_to_b64(image_path))

        parser = PydanticOutputParser(pydantic_object=RefImageIndicesAndTextPrompt)

        messages = [
            SystemMessage(content=system_prompt_template_select_reference_images_multimodal.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_prompt_template_select_reference_images.format(frame_description=frame_description, image_descriptions="\n".join(human_content)))
        ]

        chain = self.chat_model | parser

        try:
            response = await chain.ainvoke(messages)
            reference_image_path_and_text_pairs = [filtered_image_path_and_text_pairs[i] for i in response.ref_image_indices]
            return {
                "reference_image_path_and_text_pairs": reference_image_path_and_text_pairs,
                "text_prompt": response.text_prompt,
            }

        except Exception as e:
            logging.error(f"Error get image prompt: \n{e}")
            raise e


