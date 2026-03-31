import os
import logging
import cv2
from typing import List, Tuple, Union, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

from interfaces import ShotDescription, ShotBriefDescription, Camera, ImageOutput, VideoOutput


from moviepy import VideoFileClip
from PIL import Image


system_prompt_template_select_reference_camera = \
"""
[Role]
You are a professional video editing expert specializing in multi-camera shot analysis and scene structure modeling. You have deep knowledge of cinematic language, enabling you to understand shot sizes (e.g., wide shot, medium shot, close-up) and content inclusion relationships. You can infer hierarchical structures between camera positions based on corresponding shot descriptions.

[Task]
Your task is to analyze the input camera position data to construct a "camera position tree". This tree structure represents a relationship where a parent camera's content encompasses that of a child camera. Specifically, you need to identify the parent camera for each camera position (if one exists) and determine the dependent shot indices (i.e., the specific shots within the parent camera's footage that contain the child camera's content). If a camera position has no parent, output None.

[Input]
The input is a sequence of cameras. The sequence will be enclosed within <CAMERA_SEQ> and </CAMERA_SEQ>.
Each camera contains a sequence of shots filmed by the camera, which will be enclosed within <CAMERA_N> and </CAMERA_N>, where N is the index of the camera.

Below is an example of the input format:

<CAMERA_SEQ>
<CAMERA_0>
Shot 0: Medium shot of the street. Alice and Bob are walking towards each other.
Shot 2: Medium shot of the street. Alice and Bob hug each other.
</CAMERA_0>
<CAMERA_1>
Shot 1: Close-up of the Alice's face. Her expression shifts from surprise to delight as she recognizes Bob.
</CAMERA_1>
</CAMERA_SEQ>


[Output]
{format_instructions}

[Guidelines]
- The language of all output values (not include keys) should be consistent with the language of the input.
- Content Inclusion Check: The parent camera should as fully as possible contain the child camera's content in certain shots (e.g., a parent medium two-shot encompasses a child over-the-shoulder reverse shot). Analyze shot descriptions by comparing keywords (e.g., characters, actions, setting) to ensure the parent shot's field of view covers the child shot's.
- Transition Smoothness Priority: Larger shot size as parent camera is preferred, such as Wide Shot -> Medium Shot or Medium Shot -> Close-up. The shot sizes of adjacent parent and child nodes should be as similar as possible. A direct transition from a long shot to a close-up is not allowed unless absolutely necessary.
- Temporal Proximity: Each camera is described by its corresponding first shot, and the parent camera is located based on the description of the first shot. The shot index of the parent camera should be as close as possible to the first shot index of the child camera.
- Logical Consistency: The camera tree should be acyclic, avoid circular dependencies. If a camera is contained by multiple potential parents, select the best match (based on shot size and content). If there is no suitable parent camera, output None.
- When a broader perspective is not available, choose the shot with the largest overlapping field of view as the parent (the one with the most information overlap), or a shot can also serve as the parent of a reverse shot. When two cameras can be the parent of each other, choose the one with the smaller index as the parent of the camera with the larger index.
- Only one camera can exist without a parent.
- When describing the elements lost in a shot, carefully compare the details between the parent shot and the child shot. For example, the parent shot is a medium shot of Character A and Character B facing each other (both in profile to the camera), while the child shot is a close-up of Character A (with Character A facing the camera directly). In this case, the child shot lacks the frontal view information of Character A.
- The first camera must be the root of the camera tree.
"""

# [Role] 
# 你是专业的视频编辑专家，擅长多镜头镜头分析和场景结构建模。你对电影语言有深入的了解，使你能够理解镜头尺寸（例如，广角镜头，中景镜头，特写镜头）和内容包含关系。你可以根据相应的镜头描述来推断相机位置之间的层次结构。 
 
# (任务) 
# 你的任务是分析输入的摄像机位置数据来构建一个“摄像机位置树”。这个树状结构表示一种关系，其中父相机的内容包含子相机的内容。具体来说，您需要为每个摄像机位置（如果存在的话）识别父摄像机，并确定相关的镜头索引（即，父摄像机镜头中包含子摄像机内容的特定镜头）。如果一个摄像机位置没有父节点，则输出None。 
 
# (输入) 
# 输入是一系列摄像机。序列将包含在<CAMERA_SEQ>和</CAMERA_SEQ>中。 
# 每台摄像机包含一个摄像机拍摄的镜头序列，该序列将包含在<CAMERA_N>和</CAMERA_N>中，其中N为摄像机的索引。 
 
# 下面是输入格式的示例： 
 
# < CAMERA_SEQ > 
# < CAMERA_0 > 
# 镜头0：街道的中景。爱丽丝和鲍勃正走向对方。 
# 镜头2：街道的中景。爱丽丝和鲍勃互相拥抱。 
# < / CAMERA_0 > 
# < CAMERA_1 > 
# 镜头1：爱丽丝的脸部特写。当她认出鲍勃时，她的表情从惊讶变成了高兴。 
# < / CAMERA_1 > 
# < / CAMERA_SEQ > 
 
 
# (输出) 
# {format_instructions} 
 
# (指南) 
# —所有输出值（不包括键）的语言必须与输入语言保持一致。 
# -内容包含检查：在某些镜头中，父镜头应该尽可能完整地包含子镜头的内容（例如，父镜头的两张照片包含了一个孩子过肩的反向镜头）。通过比较关键词（如人物、动作、场景）来分析镜头描述，确保父镜头的视野覆盖子镜头的视野。 
# —过渡平滑优先：镜头尺寸比父相机大，如广角->中镜头或中镜头->特写。相邻父节点和子节点的镜头尺寸应尽可能相似。除非绝对必要，否则不允许从长镜头直接过渡到特写镜头。 
# —时间接近：每个相机由其对应的第一个镜头来描述，并根据第一个镜头的描述来定位父相机。父相机的拍摄索引应尽可能接近子相机的第一个拍摄索引。 
# —逻辑一致性：相机树应该是无循环的，避免循环依赖。如果一个相机包含多个潜在的父母，选择最佳匹配（基于镜头大小和内容）。如果没有合适的父摄像机，输出None。 
# -当没有更广阔的视角时，选择重叠视野最大的镜头作为父镜头（信息重叠最多的镜头），或者一个镜头也可以作为反向镜头的父镜头。当两个摄像头可以互为父摄像头时，选择索引较小的摄像头作为索引较大的摄像头的父摄像头。 
# —只有一个摄像机可以存在，没有父节点。 
# -在描述镜头中丢失的元素时，仔细比较父镜头和子镜头之间的细节。例如，父镜头是角色a和角色B面对面的中景镜头（都是对着镜头的侧面），而子镜头是角色a的特写镜头（角色a直接对着镜头）。在这种情况下，子镜头缺少角色A的正面视图信息。 
# —第一个摄像机必须是摄像机树的根节点。

human_prompt_template_select_reference_camera = \
"""
<CAMERA_SEQ>
{camera_seq_str}
</CAMERA_SEQ>
"""


class CameraParentItem(BaseModel):
    parent_cam_idx: Optional[int] = Field( 
        default=None, 
        description="The index of the parent camera. Set to None if the camera has no parent (e.g., for a root camera).",
        examples=[0, 1, None], 
    )
    parent_shot_idx: Optional[int] = Field( 
        default=None, 
        description="The index of the dependent shot. Set to None if the camera has no parent (e.g., for a root camera).",
        examples=[0, 3, None], 
    )
    reason: str = Field(
        description="The reason for the selection of the parent camera. If the camera has no parent, it should explain why it's a root camera.",
        examples=[
            "The parent shot's field of view covers the child shot's field of view (from medium shot to close-up)",
            "The parent shot and the child shot have a shot/reverse shot relationship.",
            "CAMERA_0 (Shot 0) establishes the entire scene and contains all characters and the setting. It is the root camera." # 补充 LLM 实际输出的例子
        ],
    )
    is_parent_fully_covers_child: Optional[bool] = Field( 
        default=None, 
        description="Whether the parent camera fully covers the child camera's content. Set to None if the camera has no parent.",
        examples=[True, False, None], 
    )
    missing_info: Optional[str] = Field(
        default=None,
        description="The missing elements in the child shot that are not covered by the parent shot. If the parent shot fully covers the child shot, set this to None.",
        examples=[
            "The frontal view of Alice.",
            None,
        ],
    )

class CameraTreeResponse(BaseModel):
    camera_parent_items: List[Optional[CameraParentItem]] = Field(
        description="The parent camera items for each camera. If a camera has no parent, set this to None. The length of the list should be the same as the number of cameras.",
    )



class CameraImageGenerator:

    def __init__(
        self,
        chat_model,
        image_generator,
        video_generator,
    ):
        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator


    async def construct_camera_tree(
        self,
        cameras: List[Camera],
        shot_descs: List[Union[ShotDescription, ShotBriefDescription]],
    ) -> List[Camera]:
        parser = PydanticOutputParser(pydantic_object=CameraTreeResponse)

        camera_seq_str = "<CAMERA_SEQ>\n"
        for cam in cameras:
            camera_seq_str += f"<CAMERA_{cam.idx}>\n"
            for shot_idx in cam.active_shot_idxs:
                camera_seq_str += f"Shot {shot_idx}: {shot_descs[shot_idx].visual_desc}\n"
            camera_seq_str += f"</CAMERA_{cam.idx}>\n"
        camera_seq_str += "</CAMERA_SEQ>"

        messages = [
            SystemMessage(content=system_prompt_template_select_reference_camera.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_prompt_template_select_reference_camera.format(camera_seq_str=camera_seq_str)),
        ]

        chain = self.chat_model | parser
        response: CameraTreeResponse = await chain.ainvoke(messages)
        for cam, parent_cam_item in zip(cameras, response.camera_parent_items):
            cam.parent_cam_idx = parent_cam_item.parent_cam_idx if parent_cam_item is not None else None
            cam.parent_shot_idx = parent_cam_item.parent_shot_idx if parent_cam_item is not None else None
            cam.reason = parent_cam_item.reason if parent_cam_item is not None else None
            cam.parent_shot_idx = parent_cam_item.parent_shot_idx if parent_cam_item is not None else None
            cam.is_parent_fully_covers_child = parent_cam_item.is_parent_fully_covers_child if parent_cam_item is not None else None
            cam.missing_info = parent_cam_item.missing_info if parent_cam_item is not None else None
        return cameras


    async def generate_transition_video(
        self,
        first_shot_visual_desc: str,
        second_shot_visual_desc: str,
        first_shot_ff_path: str,
    ) -> VideoOutput:

        # 两个镜头。镜头之间的过渡是切到。两张照片的风格要一致。
        prompt = f"Two shots. The transition between the shots is a cut to. The style of the two shots should be consistent."
        prompt += f"\nThe first shot description: {first_shot_visual_desc}."
        prompt += f"\nThe second shot description: {second_shot_visual_desc}."
        reference_image_paths = [first_shot_ff_path]
        video_output = await self.video_generator.generate_single_video(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
        )
        return video_output


    def get_new_camera_image(
        self,
        transition_video_path: str,
    ) -> ImageOutput:
        video = open_video(transition_video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        output_dir = os.path.join(os.path.dirname(transition_video_path), "cache")
        os.makedirs(output_dir, exist_ok=True)
        split_video_ffmpeg(transition_video_path, scene_list, output_dir, show_progress=True)


        video_name = os.path.basename(transition_video_path).split('.')[0]
        second_video_path = os.path.join(output_dir, f"{video_name}-Scene-002.mp4")
        if os.path.exists(second_video_path):
            # use first frame of second shot as new camera image
            clip = VideoFileClip(second_video_path)
            ff = clip.get_frame(0)
            ff = Image.fromarray(ff.astype('uint8'), 'RGB')
            return ImageOutput(fmt="pil", ext="png", data=ff)
        else:
            # use last frame of transition video to instead
            clip = VideoFileClip(transition_video_path)
            lf_time = clip.duration - (1 / clip.fps)
            lf_time = max(0, lf_time)
            lf = clip.get_frame(lf_time)
            lf = Image.fromarray(lf.astype('uint8'), 'RGB')
            return ImageOutput(fmt="pil", ext="png", data=lf)


    async def generate_first_frame(
        self,
        shot_desc: ShotDescription,
        character_portrait_path_and_text_pairs: List[Tuple[str, str]],
    ) -> ImageOutput:
        prompt = ""
        reference_image_paths = []
        for i,(path, text )in enumerate(character_portrait_path_and_text_pairs):
            prompt += f"Image {i}: {text}\n"
            reference_image_paths.append(path)
        prompt += f"Generate an image based on the following description: {shot_desc.ff_desc}."
        image_output = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            size="2560x1440",
        )
        return image_output
