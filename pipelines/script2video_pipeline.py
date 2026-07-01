import os
import shutil
import json
import logging
import asyncio
import time
import math
import subprocess
from typing import Optional, Dict, List, Tuple, Literal
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from agents import *
from agents.environment_agent import EnvironmentDesign, CameraCoverage
from agents.prompt_converter import ShotDescriptionWithDialogues, Dialogue
from agents.narration_agent import NarrationItem, ShotItem, Interval
import yaml
from interfaces import *
from tools.render_backend import RenderBackend
from utils.provider_presets import create_chat_model

class Script2VideoPipeline:

    # events
    character_portrait_events = {}
    shot_desc_events = {}
    frame_events = {}


    def __init__(
        self,
        chat_model: str,
        image_generator,
        video_generator,
        audio_generator,
        working_dir: str,
        interrupt_step: str = None,
        gacha_config: dict = None,
        comfyui_enable: bool = True,
    ):

        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.audio_generator = audio_generator

        self.character_extractor = CharacterExtractor(chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(image_generator=self.image_generator)
        self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model)
        self.camera_image_generator = CameraImageGenerator(chat_model=self.chat_model, image_generator=self.image_generator, video_generator=self.video_generator)
        self.reference_image_selector = ReferenceImageSelector(chat_model=self.chat_model)
        self.prompt_converter = PromptConverter()
        self.narration_agent = NarrationAgent(chat_model=self.chat_model, audio_generator=self.audio_generator)
        self.environment_designer = EnvironmentDesigner(chat_model=self.chat_model)
        self.transition_director = TransitionDirector(chat_model=self.chat_model)
        
        # 检测是否是 LTX 模型（只有 LTX 需要对话融入场景）
        self.is_ltx_model = "ltx" in type(self.video_generator).__name__.lower()

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.interrupt_step = interrupt_step
        self.gacha_config = gacha_config
        self.comfyui_enable = comfyui_enable


    @classmethod
    def init_from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        chat_model = create_chat_model(**config["chat_model"]["init_args"])
        backend = RenderBackend.from_config(config)

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config["working_dir"],
        )
        
    def check_interrupt(self, step_name: str):
        """检查是否中断"""
        
        return True if self.interrupt_step ==  step_name else False


    async def __call__(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    ):
        if characters is None:
            characters = await self.extract_characters(script=script)

            # characters_path = os.path.join(self.working_dir, "characters.json")
            # if os.path.exists(characters_path):
            #     with open(characters_path, "r", encoding="utf-8") as f:
            #         characters = [CharacterInScene.model_validate(c) for c in json.load(f)]
            #     print(f"🚀 Loaded {len(characters)} characters from existing file.")
            # else:
            #     print(f"🔍 Extracting characters from script...")
            #     characters = await self.extract_characters(script=script)
            #     with open(characters_path, "w", encoding="utf-8") as f:
            #         json.dump([c.model_dump() for c in characters], f, ensure_ascii=False, indent=4)
            #     print(f"☑️ Extracted {len(characters)} characters from script and saved to {characters_path}.")

        if character_portraits_registry is None:
            character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, "r", encoding="utf-8") as f:
                    character_portraits_registry = json.load(f)
                print(f"🚀 Loaded {len(character_portraits_registry)} character portraits from existing file.")
            else:
                print(f"🔍 Generating character portraits...")
                character_portraits_registry = await self.generate_character_portraits(
                    characters=characters,
                    character_portraits_registry=None,
                    style=style,
                )

                with open(character_portraits_registry_path, "w", encoding="utf-8") as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)
                print(f"☑️ Generated {len(character_portraits_registry)} character portraits and saved to {character_portraits_registry_path}.")

        # design environment
        environment_design = await self.design_environment(
            script=script,
            style=style,
        )

        if self.check_interrupt("environment"):
            return

        # generate environment image
        environments = await self.generate_environment_images(
            environment_design=environment_design,
            style=style,
        )
        # environments = {}
        
        if self.check_interrupt("environment_images"):
            return
        

        # design shots
        # 场景分镜
        # （电影或电视的）剧情梗概系列图片，情节串联图板
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            environment=environment_design
        )
        
        if self.check_interrupt("storyboard"):
            return

        # 分解镜头，生成首尾帧和过渡动作帧
        # decompose visual descriptions of shots
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
        )
        
        if self.check_interrupt("shot_description"):
            return

        # 构建相机树，父子级关系
        # construct camera tree
        # camera_tree = await self.construct_camera_tree(
        #     shot_descriptions=shot_descriptions,
        # )
        
        if self.check_interrupt("camera_tree"):
            return
        
        if self.gacha_config:
            # 抽卡模式
            if self.gacha_config["type"] != "video":
                shot_description = shot_descriptions[self.gacha_config["shot"]]
                # camera = camera_tree[shot_description.cam_idx]
                # first_shot_idx = camera.active_shot_idxs[0]
                # first_shot_ff_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame.png")
                shot_idx = shot_description.idx
                
                print(f"🔍Gacha Mode:: Start generating {self.gacha_config['type']} for shot {shot_idx}...")
                
                await self.generate_frame_for_single_shot(
                    shot=shot_description,
                    shot_idx=shot_idx, 
                    frame_type=self.gacha_config["type"], 
                    first_shot_ff_path_and_text_pair=(os.path.join(self.working_dir, "shots", f"{shot_idx}", "first_frame.png"), shot_description.ff_desc) if self.gacha_config["type"] != "first_frame" else None,
                    frame_desc=shot_description.ff_desc if self.gacha_config["type"] == "first_frame" else shot_description.lf_desc,
                    visible_characters=[characters[idx] for idx in (shot_description.ff_vis_char_idxs if self.gacha_config["type"] == "first_frame" else shot_description.lf_vis_char_idxs)],
                    character_portraits_registry=character_portraits_registry,
                    environment=environment_design,
                    environments_registry=environments,
                    style=style,
                )
                return
            else:
                print(f"🔍Gacha Mode:: Start generating video for shot {self.gacha_config['shot']}...")
                shot_description = shot_descriptions[self.gacha_config["shot"]]
                await self.generate_video_for_single_shot(
                    shot_description=shot_description,
                    characters=characters,
                )
        else:
            # 优先拍摄父相机？
            # 基于机位生成对应镜头的首尾帧
            # priority_shot_idxs = [camera.parent_cam_idx for camera in camera_tree if camera.parent_cam_idx is not None]
            # tasks = [
            #     self.generate_frames_for_single_camera(
            #         camera=camera,
            #         shot_descriptions=shot_descriptions,
            #         characters=characters,
            #         character_portraits_registry=character_portraits_registry,
            #         environments_registry=environments,
            #         priority_shot_idxs=priority_shot_idxs,
            #         style=style,
            #     )
            #     for camera in camera_tree
            # ]
            
            tasks = []

            for shot in shot_descriptions:
                # if shot.idx != 0:
                #     continue

                tasks.append(
                    self.generate_frame_for_single_shot(
                        shot=shot,
                        shot_idx=shot.idx,
                        frame_type="first_frame",
                        frame_desc=shot.ff_desc,
                        visible_characters=[characters[idx] for idx in shot.ff_vis_char_idxs],
                        character_portraits_registry=character_portraits_registry,
                        environment=environment_design,
                        environments_registry=environments,
                        style=style,
                    )
                )

                if shot.variation_type in ["medium", "large"]:
                    tasks.append(
                        self.generate_frame_for_single_shot(
                            shot=shot,
                            shot_idx=shot.idx,
                            frame_type="last_frame",
                            first_shot_ff_path_and_text_pair=(os.path.join(self.working_dir, "shots", f"{shot.idx}", "first_frame.png"), shot.ff_desc),
                            frame_desc=shot.lf_desc,
                            visible_characters=[characters[idx] for idx in shot.lf_vis_char_idxs],
                            character_portraits_registry=character_portraits_registry,
                            environment=environment_design,
                            environments_registry=environments,
                            style=style,
                        )
                    )

            await asyncio.gather(*tasks)

            if self.check_interrupt("camera_frame"):
                return

            # 基于首尾帧生成镜头视频
            if self.comfyui_enable:
                video_tasks = [
                    self.generate_video_for_single_shot(
                        shot_description=shot_description,
                        characters=characters,
                    )
                    for shot_description in shot_descriptions
                ]
                await asyncio.gather(*video_tasks)

            if self.check_interrupt("shot_video"):
                return

            # latent continuity transition
            shot_transitions = await self.generate_shot_transitions(
                shot_descriptions=shot_descriptions,
                style=style,
            )

            if self.check_interrupt("shot_transition"):
                return
            
            # 合并场景视频
            final_video_path = os.path.join(self.working_dir, "final_video.mp4")
            if os.path.exists(final_video_path):
                print(f"🚀 Skipped concatenating videos, already exists.")
            else:
                print(f"🎬 Starting concatenating videos...")
                story_video_path = os.path.join(self.working_dir, "story_video.mp4")
                
                if os.path.exists(story_video_path):
                    print(f"🚀 Skipped concatenating story video, already exists.")
                else:
                    print(f"🎬 Starting concatenating story video...")
                    video_clips = []
                    for idx, shot_description in enumerate(shot_descriptions):
                        video_clips.append(
                            VideoFileClip(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4"), audio=True)
                        )
                        if idx < len(shot_transitions) and shot_transitions[idx] is not None:
                            trans_path, _ = shot_transitions[idx]
                            video_clips.append(VideoFileClip(trans_path, audio=False))

                    story_video = concatenate_videoclips(video_clips, method="compose")
                    story_video.write_videofile(story_video_path, codec="libx264", preset="medium", audio_codec="aac", fps=None, audio_bitrate="192k")
                    print(f"☑️ Concatenated story video, saved to {story_video_path}.")
                
                # 生成旁白
                narration_audio_path = await self.generate_narration_audio(
                    script=script,
                    user_requirement=user_requirement,
                    storyboard=storyboard,
                    shot_transitions=shot_transitions,
                )
                
                if not narration_audio_path:
                    return
                
                # 合并视频和旁白
                print(f"🎬 Starting merging video and narration...")
                
                # 使用 ffmpeg 混合视频和旁白音频
                # 如果视频有原始音频（对话），将对话和旁白混合
                # 否则直接添加旁白音频
                # 首先检查视频是否有音频轨道
                check_audio_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "a", 
                    "-show_entries", "stream=codec_type", 
                    "-of", "csv=p=0", story_video_path
                ]
                has_audio = subprocess.run(check_audio_cmd, capture_output=True, text=True).stdout.strip()

                if has_audio:
                    # 视频有音频，混合对话和旁白
                    # 保持旁白和对话音量一致
                    subprocess.run([
                        "ffmpeg", "-y", "-i", story_video_path, "-i", narration_audio_path,
                        "-filter_complex", "[1:a]volume=1.0[narration];[0:a]volume=1.0[dialogue];[dialogue][narration]amix=inputs=2:duration=first[mixed];[mixed]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[out]",
                        "-map", "0:v", "-map", "[out]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        final_video_path
                    ], check=True, capture_output=True)
                else:
                    # 视频无音频，直接添加旁白
                    subprocess.run([
                        "ffmpeg", "-y", "-i", story_video_path, "-i", narration_audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest",
                        final_video_path
                    ], check=True, capture_output=True)
                
                print(f"✅ Merged video and narration, saved to {final_video_path}.")

            return final_video_path


    async def generate_frames_for_single_camera(
        self,
        camera: Camera,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        environments_registry: Dict[str, Dict[str, str]],
        priority_shot_idxs: List[int],
        style: str,
    ):
        # 1. generate the first_frame of the first shot of the camera
        first_shot_idx = camera.active_shot_idxs[0]
        first_shot_ff_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame.png")

        if os.path.exists(first_shot_ff_path):
            print(f"🚀 Skipped generating first_frame for shot {first_shot_idx}, already exists.")
            self.frame_events[first_shot_idx]["first_frame"].set()

        else:
            print(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            available_image_path_and_text_pairs = []

            for character_idx in shot_descriptions[first_shot_idx].ff_vis_char_idxs:
                identifier_in_scene = characters[character_idx].identifier_in_scene
                registry_item = character_portraits_registry[identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))
            
            # generate the first_frame based on the shot_description.ff_desc
            if camera.parent_shot_idx is not None:
                # generate the first_frame based on the transition video
                parent_shot_idx = camera.parent_shot_idx
                parent_shot = shot_descriptions[parent_shot_idx]
                variation_type = parent_shot.variation_type
                
                if variation_type in ["medium", "large"]:
                    await self.frame_events[parent_shot_idx]["last_frame"].wait()
                    parent_shot_ff_path = os.path.join(self.working_dir, "shots", f"{parent_shot_idx}", "last_frame.png")
                else:
                    await self.frame_events[parent_shot_idx]["first_frame"].wait()
                    parent_shot_ff_path = os.path.join(self.working_dir, "shots", f"{parent_shot_idx}", "first_frame.png")
                
                transition_video_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"transition_video_from_shot_{parent_shot_idx}.mp4")

                if os.path.exists(transition_video_path):
                    print(f"🚀 Skipped generating transition video for shot {first_shot_idx} from shot {parent_shot_idx}, already exists.")
                else:
                    print(f"🖼️ Starting transition video generation for shot {first_shot_idx} from shot {parent_shot_idx}...")
                    # 基于父级的首帧生成视频。从父级到首帧描述的过渡？
                    
                    transition_video_output = await self.camera_image_generator.generate_transition_video(
                        first_shot_visual_desc=parent_shot.visual_desc,
                        second_shot_visual_desc=shot_descriptions[first_shot_idx].visual_desc,
                        first_shot_ff_path=parent_shot_ff_path,
                    )
                    transition_video_output.save(transition_video_path)
                    print(f"☑️ Generated transition video for shot {first_shot_idx} from shot {parent_shot_idx}, saved to {transition_video_path}.")

                # 从视频中截取场景切换帧作为参考，这样能提高一致性
                new_camera_image_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"new_camera_{camera.idx}.png")
                if os.path.exists(new_camera_image_path):
                    print(f"🚀 Skipped generating new camera image for shot {first_shot_idx}, already exists.")
                else:
                    print(f"🖼️ Starting new camera image generation for shot {first_shot_idx}...")
                    new_camera_image = self.camera_image_generator.get_new_camera_image(transition_video_path)
                    new_camera_image.save(new_camera_image_path)
                    print(f"☑️ Generated new camera image for shot {first_shot_idx} (not completed), saved to {new_camera_image_path}.")
                    
                if new_camera_image_path:
                    available_image_path_and_text_pairs.append(
                        (
                            new_camera_image_path,
                            f"The composition and background are correct but some elements may be wrong. The wrong elements should be replaced.Wrong elements: {camera.missing_info}.You must select this image as the main reference and replace the characters in the image with the provided character portraits. Don't change the background."
                            # 构图和背景是正确的，但有些元素可能是错误的。错误的元素应该被替换。{camera.missing_info}。您必须选择此图像作为主要参考，并将图像中的字符替换为提供的字符肖像。不要改变背景。
                        )
                    )


            # 如果有父镜头，并且父镜头完全覆盖了子镜头；则直接使用切换场景作为首帧图；否则生成首帧图
            # if camera.parent_shot_idx is None or camera.missing_info is not None:
            ff_image: ImageOutput = await self.generate_image_for_single_shot(
                shot_idx=first_shot_idx,
                frame_type="first_frame",
                frame_desc=shot_descriptions[first_shot_idx].ff_desc,
                available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                style=style
            )
            ff_image.save(first_shot_ff_path)
            self.frame_events[first_shot_idx]["first_frame"].set()
            print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
            # else:
            #     shutil.copy(new_camera_image_path, first_shot_ff_path)
            #     self.frame_events[first_shot_idx]["first_frame"].set()
            #     print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}. Parent shot fully covers the child shot, direct use.")


        # 2. generate the following frames of the camera
        priority_tasks = []
        normal_tasks = []

        # 如果是中大型镜头，生成尾帧
        if shot_descriptions[first_shot_idx].variation_type in ["medium", "large"]:
            task = self.generate_frame_for_single_shot(
                shot_idx=first_shot_idx, 
                frame_type="last_frame", 
                first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                frame_desc=shot_descriptions[first_shot_idx].lf_desc,
                visible_characters=[characters[idx] for idx in shot_descriptions[first_shot_idx].lf_vis_char_idxs],
                character_portraits_registry=character_portraits_registry,
                style=style,
            )
            normal_tasks.append(task)

        # 生成后续镜头。同一个机位的后续图片参考第一个镜头图片
        for shot_idx in camera.active_shot_idxs[1:]:
            # 基于首镜头生成后续镜头首帧
            first_frame_task = self.generate_frame_for_single_shot(
                shot_idx=shot_idx, 
                frame_type="first_frame", 
                first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                frame_desc=shot_descriptions[shot_idx].ff_desc,
                visible_characters=[characters[idx] for idx in shot_descriptions[shot_idx].ff_vis_char_idxs],
                character_portraits_registry=character_portraits_registry,
                style=style,
            )
            if shot_idx in priority_shot_idxs:
                priority_tasks.append(first_frame_task)
            else:
                normal_tasks.append(first_frame_task)


            # 基于首镜头生成后续镜头尾帧
            if shot_descriptions[shot_idx].variation_type in ["medium", "large"]:
                last_frame_task = self.generate_frame_for_single_shot(
                    shot_idx=shot_idx, 
                    frame_type="last_frame", 
                    first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                    frame_desc=shot_descriptions[shot_idx].lf_desc,
                    visible_characters=[characters[idx] for idx in shot_descriptions[shot_idx].lf_vis_char_idxs],
                    character_portraits_registry=character_portraits_registry,
                    style=style,
                )
                normal_tasks.append(last_frame_task)


        await asyncio.gather(*priority_tasks)
        await asyncio.gather(*normal_tasks)



    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
        characters: List[CharacterInScene],
    ):
        video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
        if os.path.exists(video_path):
            print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
        else:
            if not self.gacha_config:
                await self.frame_events[shot_description.idx]["first_frame"].wait()
                if shot_description.variation_type in ["medium", "large"]:
                    await self.frame_events[shot_description.idx]["last_frame"].wait()

            frame_paths = []
            frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "first_frame.png"))
            if shot_description.variation_type in ["medium", "large"]:
                frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png"))

            print(f"🎬 Starting video generation for shot {shot_description.idx}...")
            
            # 生成融入音效的场景视觉描述和对话列表。对话列表后续做旁白的时候需要
            # LTX 模型：对话必须融入场景叙述中，而不是分开展示
            # 将 motion_desc 和 audio_desc 融合，使对话出现在正确的动作位置
            final_prompt_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "ltx_prompt.json")
            if os.path.exists(final_prompt_path):
                print(f"🚀 Skipped generating LTX prompt for shot {shot_description.idx}, already exists.")
                with open(final_prompt_path, "r", encoding="utf-8") as f:
                    shot_description_with_dialogues: ShotDescriptionWithDialogues = ShotDescriptionWithDialogues.model_validate(json.load(f))
            else:
                shot_description_with_dialogues: ShotDescriptionWithDialogues = await self.prompt_converter.convert(
                    audio_desc=shot_description.audio_desc,
                    # visual_desc=shot_description.visual_desc,
                    motion_desc=shot_description.motion_desc,
                    characters=characters,
                    shot_duration=shot_description.shot_duration or 5.0,
                )
                # 写入文件
                with open(final_prompt_path, 'w', encoding='utf-8') as f:
                    json.dump(shot_description_with_dialogues.model_dump(), f, ensure_ascii=False, indent=4)
            
            # final_prompt = shot_description_with_dialogues.prompt
            dialogue_audio_path = None

            if self.is_ltx_model:
                final_prompt = shot_description_with_dialogues.prompt
                print(f"📝 LTX Prompt: {final_prompt[:100]}...")
                
                # 生成音频
                if shot_description_with_dialogues.dialogues:
                    parent_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}")
                    dialogue_audio_path = os.path.join(parent_path, "final_dialogues.flac")
                    
                    if os.path.exists(dialogue_audio_path):
                        print(f"🚀 Skipped generating dialogue audio for shot {shot_description.idx}, already exists.")
                    else:
                        for idx, dialogue in enumerate(shot_description_with_dialogues.dialogues):
                            audio_output = await self.audio_generator.generate_single_audio(
                                prompt=dialogue.dialogue,
                                character=dialogue.speaker,
                                gender=dialogue.gender,
                            )
                            audio_output.save(os.path.join(parent_path, f"{idx}.flac"))
                            
                        dialogue_audio_path, dialogues = await self.audio_generator.adjust_audio(
                            duration=shot_description.shot_duration or 5.0,
                            dialogues=shot_description_with_dialogues.dialogues,
                            shot_path=parent_path,
                        )
                        
                        # 重新写入final_prompt_path，更新dialogues
                        with open(final_prompt_path, 'w', encoding='utf-8') as f:
                            shot_description_with_dialogues.dialogues = dialogues
                            json.dump(shot_description_with_dialogues.model_dump(), f, ensure_ascii=False, indent=4)
            else:
                # 非 LTX 模型：保持 motion_desc 和 audio_desc 分离的原始格式
                final_prompt = f"{shot_description.motion_desc}\n{shot_description.audio_desc}"
            
            # 基于运镜生成视频
            video_output = await self.video_generator.generate_single_video(
                prompt=final_prompt + "\n整个过程光影流动自然，人物动作与物体形变连贯衔接。",
                reference_image_paths=frame_paths,
                audio_path=dialogue_audio_path,
                duration=int(shot_description.shot_duration or 5.0),
                use_xianxia_lora=shot_description_with_dialogues.use_xianxia_lora,
                is_small_people=shot_description_with_dialogues.is_small_people,
            )
            
            # 后续统一成 1680
            if shot_description_with_dialogues.is_small_people:
                self._scale_video(video_output, video_path)
            else:
                video_output.save(video_path)

            if shot_description.variation_type == "small":
                # 小镜头，提取最后一帧
                last_frame_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png")

                if os.path.exists(video_path):
                    subprocess.run([
                        "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
                        "-update", "1", "-q:v", "1", last_frame_path,
                    ], check=True, capture_output=True)

            print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")

    async def generate_frame_for_single_shot(
        self,
        shot: ShotDescription,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        *,
        first_shot_ff_path_and_text_pair: Optional[Tuple[str, str]] = None,
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        environment: EnvironmentDesign,
        environments_registry: Dict[str, Dict[str, str]],
        style: str,
    ) -> ImageOutput:

        frame_image_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}.png")

        if os.path.exists(frame_image_path):
            print(f"🚀 Skipped generating {frame_type} for shot {shot_idx}, already exists.")

        else:
            print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")
            available_image_path_and_text_pairs = []
            for visible_character in visible_characters:
                identifier_in_scene = visible_character.identifier_in_scene
                registry_item = character_portraits_registry[identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))

            if first_shot_ff_path_and_text_pair:
                available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)

            # enviroment available
            if shot.camera_id:
                environment = environments_registry[shot.camera_id]
                available_image_path_and_text_pairs.append((environment["path"], environment["description"]))
            else:
                # 根据 new_camera_reason 生成新的环境图
                if shot.new_camera_reason:
                    if not self.comfyui_enable:
                        return

                    new_camera = await self.environment_designer.design_new_camera(
                        environment=environment,
                        shot_description=shot.visual_desc,
                        new_camera_reason=shot.new_camera_reason,
                        style=style,
                    )
                    new_camera_image_path = await self.generate_environment_image(new_camera, style)
                    available_image_path_and_text_pairs.append((new_camera_image_path, f"{new_camera.lens_type}，{new_camera.composition_style}，{new_camera.visible_environment_area}"))

            frame_image: ImageOutput = await self.generate_image_for_single_shot(
                shot_idx=shot_idx,
                frame_type=frame_type,
                frame_desc=frame_desc,
                available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                style=style
            )
            frame_image.save(frame_image_path)
            print(f"☑️ Generated {frame_type} frame for shot {shot_idx}, saved to {frame_image_path}.")

        self.frame_events[shot_idx][frame_type].set()
        return frame_image_path


    async def generate_image_for_single_shot(
        self,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        frame_desc: str,
        available_image_path_and_text_pairs: List[Tuple[str, str]],
        style: str,
    ) -> ImageOutput:
        """
        生成镜头图片
        """
        
        selector_output_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_selector_output.json")
        if os.path.exists(selector_output_path):
            with open(selector_output_path, 'r', encoding='utf-8') as f:
                selector_output = json.load(f)
            print(f"🚀 Loaded existing reference image selection and prompt for {frame_type} frame of shot {shot_idx} from {selector_output_path}.")
        else:
            print(f"🔍 Selecting reference images and generating prompt for {frame_type} frame of shot {shot_idx}...")
            # 基于镜头描述和参考图像，生成一个描述要创建的图像的文本提示符，指定生成的图像中的哪些元素应该引用哪个图像描述（以及其中的哪些元素）
            # 参考人物特征 + 前面帧，保证角色、环境、风格一致性
            # 这里只参考了人物肖像
            selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                frame_description=frame_desc,
                only_text_model=True,
            )
            with open(selector_output_path, 'w', encoding='utf-8') as f:
                json.dump(selector_output, f, ensure_ascii=False, indent=4)
            print(f"☑️ Selected reference images and generated prompt for {frame_type} frame of shot {shot_idx}, saved to {selector_output_path}.")
        
        if not selector_output:
            return

        if not self.comfyui_enable:
            return

        reference_image_path_and_text_pairs, prompt = selector_output["reference_image_path_and_text_pairs"], selector_output["text_prompt"]
        prefix_prompt = f"style: {style}\n"
        # for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
        #     prefix_prompt += f"Image {i}: {text}\n"
        prompt = f"{prefix_prompt}\n{prompt}"
        reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

        # 检查 reference_image_paths，如果有引用非同一个 shot index 的首尾帧，则 wait
        if not self.gacha_config:
            for reference_image_path in reference_image_paths:
                ref_frame_type = None
                for ft in ["first_frame", "last_frame"]:
                    if ft in reference_image_path:
                        ref_frame_type = ft
                        break

                if ref_frame_type:
                    parts = reference_image_path.split("/")
                    shots_idx = parts.index("shots") if "shots" in parts else -1
                    frame_index = int(parts[shots_idx + 1]) if shots_idx >= 0 else None

                    if frame_index is not None:
                        await self.frame_events[frame_index][ref_frame_type].wait()

        frame_image: ImageOutput = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            size="2560x1440",
        )
        
        # TODO:: 评估、反馈、循环
        
        
        return frame_image


    async def construct_camera_tree(
        self,
        shot_descriptions: List[ShotDescription],
    ):
        camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")

        if os.path.exists(camera_tree_path):
            with open(camera_tree_path, "r", encoding="utf-8") as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            print(f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
            return camera_tree

        cameras: List[Camera] = []
        for shot_description in shot_descriptions:
            if shot_description.cam_idx not in [camera.idx for camera in cameras]:
                cameras.append(Camera(idx=shot_description.cam_idx, active_shot_idxs=[shot_description.idx]))
            else:
                cameras[shot_description.cam_idx].active_shot_idxs.append(shot_description.idx)

        camera_tree = await self.camera_image_generator.construct_camera_tree(cameras=cameras, shot_descs=shot_descriptions)
        with open(camera_tree_path, "w", encoding="utf-8") as f:
            json.dump([camera.model_dump() for camera in camera_tree], f, ensure_ascii=False, indent=4)
            
        print(f"✅ Constructed camera tree and saved to {camera_tree_path}.")
        return camera_tree

    async def extract_characters(
        self,
        script: str,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(character) for character in characters]
            print(f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            print(f"✅ Extracted {len(characters)} characters from script and saved to {save_path}.")

        for character in characters:
            self.character_portrait_events[character.idx] = asyncio.Event()

        return characters


    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
    ):
        character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}


        tasks = [
            self.generate_portraits_for_single_character(character, style)
            for character in characters
            if character.identifier_in_scene not in character_portraits_registry
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)

            print(f"✅ Completed character portrait generation for {len(characters)} characters.")
        else:
            print("🚀 All characters already have portraits, skipping portrait generation.")
        return character_portraits_registry


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
    ):
        character_dir = os.path.join(self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)


        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path):
            pass
        else:
            side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
            side_portrait_output.save(side_portrait_path)

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path):
            pass
        else:
            back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
            back_portrait_output.save(back_portrait_path)

        self.character_portrait_events[character.idx].set()

        print(f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")

        return {
            character.identifier_in_scene: {
                "front": {
                    "path": front_portrait_path,
                    "description": f"A front view portrait of {character.identifier_in_scene}.",
                },
                "side": {
                    "path": side_portrait_path,
                    "description": f"A side view portrait of {character.identifier_in_scene}.",
                },
                "back": {
                    "path": back_portrait_path,
                    "description": f"A back view portrait of {character.identifier_in_scene}.",
                },
            }
        }



    async def design_environment(
        self,
        script: str,
        style: str,
    ) -> EnvironmentDesign:
        environment_design_path = os.path.join(self.working_dir, "environment_design.json")
        if os.path.exists(environment_design_path):
            with open(environment_design_path, 'r', encoding='utf-8') as f:
                environment_design = json.load(f)
            environment_design = EnvironmentDesign.model_validate(environment_design)
            print(f"🚀 Loaded environment design from existing file.")
        else:
            print(f"🎬 Designing environment...")
            environment_design = await self.environment_designer.design_environment(
                scene_description=script,
                style=style,
                retry_timeout=300,
            )
            with open(environment_design_path, 'w', encoding='utf-8') as f:
                json.dump(environment_design.model_dump(), f, ensure_ascii=False, indent=4)
            print(f"✅ Environment designed and saved to {environment_design_path}.")

        return environment_design


    async def generate_environment_images(
        self,
        environment_design: EnvironmentDesign,
        style: str,
    ) -> Dict[str, Dict[str, str]]:
        """生成环境母图和各机位图，返回 camera_id → 图片路径 映射"""
        env_images_dir = os.path.join(self.working_dir, "environments")
        os.makedirs(env_images_dir, exist_ok=True)

        # 1. 生成母图
        master_image_path = os.path.join(env_images_dir, "master.png")
        if os.path.exists(master_image_path):
            print(f"🚀 Loaded existing master environment image.")
        else:
            if not self.comfyui_enable:
                return {}

            master_prompt = f"style: {style}\n{environment_design.master_prompt}" if style else environment_design.master_prompt
            print(f"🎬 Generating master environment image...")
            master_image: ImageOutput = await self.image_generator.generate_single_image(
                prompt=master_prompt,
                size="2560x1440",
            )
            master_image.save(master_image_path)
            print(f"✅ Master environment image saved to {master_image_path}.")

        # 2. 基于母图生成各机位图
        camera_image_map: Dict[str, str] = {}
        for cam_coverage in environment_design.camera_coverages:
            cam_image_path = await self.generate_environment_image(cam_coverage, style)

            if cam_image_path:
                camera_image_map[cam_coverage.camera_id] = {
                    "path": cam_image_path,
                    "description": f"{cam_coverage.lens_type}，{cam_coverage.composition_style}，{cam_coverage.visible_environment_area}",
                }
                print(f"✅ Camera image for {cam_coverage.camera_id} saved to {cam_image_path}.")

        # 3. 保存映射
        camera_image_map_path = os.path.join(self.working_dir, "camera_image_map.json")

        if os.path.exists(camera_image_map_path):
            with open(camera_image_map_path, 'r', encoding='utf-8') as f:
                camera_image_map = json.load(f)
            print(f"🚀 Loaded camera → image map from existing file.")
        else:
            with open(camera_image_map_path, 'w', encoding='utf-8') as f:
                json.dump(camera_image_map, f, ensure_ascii=False, indent=4)
            print(f"✅ Camera → image map saved to {camera_image_map_path}.")

        return camera_image_map

    async def generate_environment_image(
        self,
        camera_coverage: CameraCoverage,
        style: str,
    ) -> str:
        working_dir = os.path.join(self.working_dir, "environments")
        master_image_path = os.path.join(working_dir, "master.png")
        cam_image_path = os.path.join(working_dir, f"{camera_coverage.camera_id}.png")
        if os.path.exists(cam_image_path):
            print(f"🚀 Loaded existing camera image for {camera_coverage.camera_id}.")
        else:
            if not self.comfyui_enable:
                return ""

            # edit_prompt = f"style: {style}\n{camera_coverage.qwen_edit_camera_prompt}" if style else camera_coverage.qwen_edit_camera_prompt
            print(f"📷 Generating camera image for {camera_coverage.camera_id} ({camera_coverage.camera_name})...")
            cam_image: ImageOutput = await self.image_generator.generate_single_image(
                prompt=camera_coverage.qwen_edit_camera_prompt,
                reference_image_paths=[master_image_path, master_image_path],
                size="2560x1440",
                workflow_name="qwen_edit_camera",
            )
            cam_image.save(cam_image_path)

        return cam_image_path

    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        environment: EnvironmentDesign,
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            print(f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            print(f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.design_storyboard(
                script=script,
                characters=characters,
                environment=environment,
                user_requirement=user_requirement,
                retry_timeout=300,
            )
            with open(storyboard_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            print(f"✅ Designed storyboard and saved to {storyboard_path}.")
            
        
        if self.check_interrupt("storyboard"):
            return storyboard

        for shot_brief_description in storyboard:
            self.shot_desc_events[shot_brief_description.idx] = asyncio.Event()

        return storyboard



    async def decompose_visual_descriptions(
        self,
        shot_brief_descriptions: List[ShotBriefDescription],
        characters: List[CharacterInScene],
    ):
        tasks = [
            self.decompose_visual_description_for_single_shot_brief_description(
                shot_brief_description,
                characters
            )
            for shot_brief_description in shot_brief_descriptions
        ]

        shot_descriptions = await asyncio.gather(*tasks)
        return shot_descriptions


    async def decompose_visual_description_for_single_shot_brief_description(
        self,
        shot_brief_description: ShotBriefDescription,
        characters: List[CharacterInScene],
    ):
        shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_brief_description.idx}", "shot_description.json")
        os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)

        if os.path.exists(shot_description_path):
            with open(shot_description_path, 'r', encoding='utf-8') as f:
                shot_description = ShotDescription.model_validate(json.load(f))
            print(f"🚀 Loaded shot {shot_brief_description.idx} description from existing file.")
        else:
            shot_description = await self.storyboard_artist.decompose_visual_description(
                shot_brief_desc=shot_brief_description,
                characters=characters,
                retry_timeout=300,
            )
            with open(shot_description_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            print(f"✅ Decomposed visual description for shot {shot_brief_description.idx} and saved to {shot_description_path}.")

        self.shot_desc_events[shot_brief_description.idx].set()
        
        if self.check_interrupt("shot_description"):
            return shot_description
        
        if shot_description.variation_type in ["medium", "large"]:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
                "last_frame": asyncio.Event(),
            }
        else:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
            }

        return shot_description

    async def generate_narration_audio(
        self,
        script: str,
        user_requirement: str,
        storyboard: List[ShotBriefDescription],
        shot_transitions: List[Optional[Tuple[str, float]]] = None,
    ):
        """
        生成旁白音频
        使用 NarrationAgent 生成旁白音频。如果有转场，将转场时长计入对应 shot 的尾部静音，
        保证旁白音频总时长 = 最终视频总时长（转场视频不加旁白）。
        """
        narration_audio_path = os.path.join(self.working_dir, "narration_audio.flac")
        narration_desc_path = os.path.join(self.working_dir, "narration_desc.json")

        if os.path.exists(narration_audio_path):
            print(f"🚀 Skipped generating narration audio, already exists.")
            return narration_audio_path
        
        print(f"🎬 Starting generating narration audio...")
        
        # 读取或生成旁白描述
        if os.path.exists(narration_desc_path):
            with open(narration_desc_path, 'r', encoding='utf-8') as f:
                narration_data = json.load(f)
            narration_desc = [NarrationItem.model_validate(item) for item in narration_data]
            print(f"🚀 Loaded narration description from existing file.")
        else:
            print(f"🔍 Generating narration description...")
            
            # 1. 从每个 shot 的 ltx_prompt.json 取 dialogues，并构建 storyboard
            shot_items: List[ShotItem] = []
            for shot in storyboard:
                # 读取 ltx_prompt.json 获取 dialogues 和 shot_duration
                shot_path = os.path.join(self.working_dir, "shots", str(shot.idx))
                ltx_prompt_path = os.path.join(shot_path, "ltx_prompt.json")
                dialogues = []
                shot_duration = 5.0
                
                if os.path.exists(ltx_prompt_path):
                    with open(ltx_prompt_path, 'r', encoding='utf-8') as f:
                        print(f"Reading ltx_prompt.json for shot {shot.idx}")
                        ltx_prompt = json.load(f)
                        if ltx_prompt.get("dialogues"):
                            dialogues = [Dialogue.model_validate(d) for d in ltx_prompt["dialogues"]]
                        # 从 ltx_prompt 获取 shot_duration（如果存在）
                        if ltx_prompt.get("shot_duration"):
                            shot_duration = float(ltx_prompt["shot_duration"])
                
                # 构建 dialogue_intervals（对话占用的时间区间）
                dialogue_intervals = []
                for d in dialogues:
                    dialogue_intervals.append(Interval(
                        start=d.start_time,
                        end=d.start_time + d.duration
                    ))
                
                # 计算 available_intervals（扣除对话和0.5秒安全间隙后的可用区间）
                available_intervals = []
                
                if not dialogue_intervals:
                    # 没有对话时，整个镜头时长都可作为可用区间
                    available_intervals.append(Interval(
                        start=0,
                        end=shot_duration
                    ))
                else:
                    # 按时间排序对话区间
                    sorted_intervals = sorted(dialogue_intervals, key=lambda x: x.start)
                    
                    # 判断大于3s才算可用区间
                    if sorted_intervals[0].start > 3:
                        available_intervals.append(Interval(
                            start=0,
                            end=sorted_intervals[0].start - 3
                        ))
                    
                    for i in range(len(sorted_intervals) - 1):
                        if sorted_intervals[i + 1].start - sorted_intervals[i].end > 3:
                            available_intervals.append(Interval(
                                start=sorted_intervals[i].end,
                                end=sorted_intervals[i + 1].start
                            ))
                            
                    if sorted_intervals[-1].end < shot_duration - 3:
                        available_intervals.append(Interval(
                            start=sorted_intervals[-1].end,
                            end=shot_duration
                        ))
                
                shot_items.append(ShotItem(
                    idx=str(shot.idx),
                    shot_duration=shot_duration,
                    dialogue_intervals=dialogue_intervals,
                    available_intervals=available_intervals,
                    visual_desc=shot.visual_desc,
                    audio_desc=shot.audio_desc or ""
                ))

            # 2. 调用 NarrationAgent.generate_narration 生成旁白
            narration_desc = await self.narration_agent.generate_narration(
                story=script,
                storyboard=shot_items,
                user_requirement=user_requirement
            )
            
            # 保存旁白描述
            narration_data = [item.model_dump() for item in narration_desc]
            with open(narration_desc_path, 'w', encoding='utf-8') as f:
                json.dump(narration_data, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Generated narration description and saved to {narration_desc_path}.")
        
        # 3. 如果存在转场，将转场时长计入对应 shot 的尾部静音（转场不加旁白）
        if shot_transitions and len(shot_transitions) > 0:
            for i, trans in enumerate(shot_transitions):
                if trans is not None and i < len(narration_desc):
                    _, trans_dur = trans
                    narration_desc[i].shot_duration += trans_dur
            print(f"🔧 Adjusted narration shot_durations with transition gaps.")
        print(55555, narration_desc)
        # 4. 生成旁白音频（shot 尾部静音中自然包含转场时长）
        await self.narration_agent.generate_audio(
            narration=narration_desc,
            output_path=narration_audio_path,
            comfyui_enable=self.comfyui_enable,
        )

        if os.path.exists(narration_audio_path):
            print(f"✅ Generated narration audio saved to {narration_audio_path}.")
            return narration_audio_path
        
        return

    async def generate_shot_transitions(self, shot_descriptions: List[ShotDescription], style: str) -> List[Optional[Tuple[str, float]]]:
        """生成shot之间的转场视频。latent continuity transition"""

        if len(shot_descriptions) < 2:
            return []

        transition_dir = os.path.join(self.working_dir, "transitions")
        transition_desc_path = os.path.join(self.working_dir, "transitions.json")
        os.makedirs(transition_dir, exist_ok=True)

        transition_results = []
        transition_fragments = []

        if os.path.exists(transition_desc_path):
            with open(transition_desc_path, 'r', encoding='utf-8') as f:
                transition_data = json.load(f)
            transition_fragments = [TransFragment.model_validate(item) for item in transition_data]

        for i in range(len(shot_descriptions) - 1):
            prev_shot = shot_descriptions[i]
            next_shot = shot_descriptions[i + 1]

            # ── 前一个 shot 的尾帧 ──
            prev_shot_dir = os.path.join(self.working_dir, "shots", str(prev_shot.idx))
            prev_last_frame_path = os.path.join(prev_shot_dir, "last_frame.png")

            if not os.path.exists(prev_last_frame_path):
                # 如果前一个 shot 没有 last_frame，从视频中提取最后一帧
                video_path = os.path.join(prev_shot_dir, "video.mp4")
                if os.path.exists(video_path):
                    print(f"📸 Extracting last frame from shot {prev_shot.idx} video...")
                    subprocess.run([
                        "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
                        "-update", "1", "-q:v", "1", prev_last_frame_path,
                    ], check=True, capture_output=True)
                    print(f"☑️ Extracted last frame for shot {prev_shot.idx}.")
                else:
                    print(f"⚠️ No video found for shot {prev_shot.idx}, skipping transition {prev_shot.idx}->{next_shot.idx}.")
                    transition_fragments.append(None)
                    continue

            # ── 后一个 shot 的首帧 ──
            next_shot_dir = os.path.join(self.working_dir, "shots", str(next_shot.idx))
            next_first_frame_path = os.path.join(next_shot_dir, "first_frame.png")

            if not os.path.exists(next_first_frame_path):
                print(f"⚠️ No first frame found for shot {next_shot.idx}, skipping transition {prev_shot.idx}->{next_shot.idx}.")
                transition_fragments.append(None)
                continue

            # ── 转场视频路径 ──
            transition_video_path = os.path.join(
                transition_dir, f"transition_{prev_shot.idx}_to_{next_shot.idx}.mp4"
            )

            if os.path.exists(transition_video_path):
                print(f"🚀 Skipped transition {prev_shot.idx}->{next_shot.idx}, already exists.")
                clip = VideoFileClip(transition_video_path)
                transition_results.append((transition_video_path, clip.duration))
                clip.close()
                continue

            if not self.comfyui_enable:
                transition_results.append(None)
                continue

            print(f"🎬 Generating latent continuity transition {prev_shot.idx} -> {next_shot.idx}...")

            # ── 调用 TransitionDirector 生成转场描述 ──
            trans_fragment = next(filter(lambda x: x.idx == i, transition_fragments), None)

            if trans_fragment is None:
                trans_fragment: TransFragment = await self.transition_director.design_transition(
                    previous_shot=prev_shot.visual_desc,
                    next_shot=next_shot.visual_desc,
                    previous_last_frame=prev_shot.lf_desc if prev_shot.lf_desc else "",
                    next_first_frame=next_shot.ff_desc if next_shot.ff_desc else "",
                    environment_style=style,
                )
                trans_fragment.idx = i
                transition_fragments.append(trans_fragment)

            # ── 使用首尾帧生成转场视频 ──
            trans_output = await self.video_generator.generate_single_video(
                prompt=trans_fragment.motion_desc,
                reference_image_paths=[prev_last_frame_path, next_first_frame_path],
                duration=trans_fragment.duration,
            )
            
            # self._scale_video(trans_output, transition_video_path)
            trans_output.save(transition_video_path)

            clip = VideoFileClip(transition_video_path)
            transition_results.append((transition_video_path, clip.duration))
            clip.close()

            print(f"☑️ Generated transition {prev_shot.idx} -> {next_shot.idx}, saved to {transition_video_path}.")

        if len(transition_fragments) > 0:
            with open(transition_desc_path, 'w', encoding='utf-8') as f:
                json.dump([t.model_dump() for t in transition_fragments], f, ensure_ascii=False, indent=4)

        transition_data = []

        for trans in transition_fragments:
            if trans.enable:
                transition_data.append(transition_results[trans.idx])
            else:
                transition_data.append(None)

        return transition_data

    def _scale_video(self, video_output: VideoOutput, video_path: str):
        """缩放视频"""
        
        temp_video_path = video_path.replace(".mp4", "_temp.mp4")
        video_output.save(temp_video_path)
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_video_path,
            "-vf", "scale=1344:768:force_original_aspect_ratio=decrease,pad=1344:768:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            video_path,
        ], check=True, capture_output=True)
        os.remove(temp_video_path)
