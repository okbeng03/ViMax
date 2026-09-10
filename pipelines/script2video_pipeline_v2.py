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
from agents.prompt_converter import ShotDescriptionWithDialogues, Dialogue
from agents.minimax_prompt.agent import PromptConverter as MinimaxPromptConverter
import yaml
from interfaces import *
from tools.render_backend import RenderBackend
from utils.provider_presets import create_chat_model
from utils.dialogues_detect import get_video_duration, detect_dialogues_from_video
from agents.minimax.storyboard import StoryboardAgent, StoryboardShot
from agents.minimax.narration import NarrationAgent as MinimaxNarrationAgent, NarrationShot

class Script2VideoPipelineV2:

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

        # self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model)
        self.storyboard_artist = StoryboardAgent(chat_model=self.chat_model)
        self.reference_image_selector = ReferenceImageSelector(chat_model=self.chat_model)
        self.minimax_prompt_converter = MinimaxPromptConverter()
        self.narration_agent = MinimaxNarrationAgent(chat_model=self.chat_model, audio_generator=self.audio_generator)
        
        # 检测是否是 minimax 模型
        self.is_minimax_model = "minimax" in type(self.video_generator).__name__.lower()

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
        # design shots
        # 场景分镜
        # （电影或电视的）剧情梗概系列图片，情节串联图板
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
        )
        
        if self.check_interrupt("intro_storyboard"):
            return

        # 分解镜头，生成首尾帧和过渡动作帧
        # decompose visual descriptions of shots
        # shot_descriptions = await self.decompose_visual_descriptions(
        #     shot_brief_descriptions=storyboard,
        #     characters=characters,
        # )
        
        # if self.check_interrupt("intro_shot_description"):
        #     return
        
        for shot_description in storyboard:
            shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "shot_description.json")
            os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)
            await self.generate_video_for_single_shot(
                shot_description=shot_description,
                characters=characters,
                style=style,
            )

        if self.check_interrupt("intro_shot_video"):
            return

        await self.generate_narration_audio(
            user_requirement=user_requirement,
            storyboard=storyboard,
        )
        
        # 合并场景视频
        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped concatenating videos, already exists.")
        else:
            print(f"🎬 Starting concatenating videos...")
            video_clips = []
            for idx, shot_description in enumerate(storyboard):
                shot_video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")

                if not os.path.exists(shot_video_path):
                    return

                video_clips.append(
                    VideoFileClip(shot_video_path, audio=True)
                )

            story_video = concatenate_videoclips(video_clips, method="compose")
            story_video.write_videofile(final_video_path, codec="libx264", preset="medium", audio_codec="aac", fps=None, audio_bitrate="192k")
            print(f"☑️ Concatenated story video, saved to {final_video_path}.")

        return final_video_path


    async def generate_video_for_single_shot(
        self,
        shot_description: StoryboardShot,
        characters: List[CharacterInScene],
        style: str,
        # prev_shot_description: Optional[ShotDescription] = None,
    ):
        # if not self.gacha_config:
        #     await self.frame_events[shot_description.idx]["first_frame"].wait()
        #     if shot_description.variation_type in ["medium", "large"]:
        #         await self.frame_events[shot_description.idx]["last_frame"].wait()

        frame_paths = []
        workflow_type = "ref"

        if shot_description.first_frame_description is not None:
            if not self.is_minimax_model:
                raise ValueError("prev_shot_description is not supported for non-minimax models.")

            # 直接用上一个镜头尾帧做当前首帧（延续性）
            frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx - 1}", "last_frame.png"))
            # shot_description.ff_desc = shot_description.first_frame_description
            # shot_description.ff_vis_char_idxs = prev_shot_description.lf_vis_char_idxs

        if shot_description.final_frame_description is not None and shot_description.workflow_type == "base":
            last_frame_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png")

            if os.path.exists(last_frame_path):
                frame_paths.append(last_frame_path)
                workflow_type = "base"
 
        # shot_description.variation_type = "small"

        if self.comfyui_enable:
            for frame_path in frame_paths:
                if not os.path.exists(frame_path):
                    raise FileNotFoundError(f"Frame {frame_path} not found!")

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
            if self.is_minimax_model:
                if workflow_type == "ref":
                    shot_description_with_dialogues: ShotDescriptionWithDialogues = await self.minimax_prompt_converter.ref_convert(
                        style=style,
                        shot_description=shot_description,
                        characters=characters,
                    )
                else:
                    shot_description_with_dialogues: ShotDescriptionWithDialogues = await self.minimax_prompt_converter.base_convert(
                        style=style,
                        shot_description=shot_description,
                        characters=characters,
                    )

            # 写入文件
            with open(final_prompt_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description_with_dialogues.model_dump(), f, ensure_ascii=False, indent=4)
        
        # final_prompt = shot_description_with_dialogues.prompt
        dialogue_audio_path = None

        if not self.comfyui_enable:
            return

        if self.is_minimax_model:
            # minimax 模型
            final_prompt = shot_description_with_dialogues.prompt
            print(f"📝 Minimax Prompt: {final_prompt[:100]}...")
        else:
            # 非 LTX 模型：保持 motion_desc 和 audio_desc 分离的原始格式
            final_prompt = f"{shot_description.motion_desc}\n{shot_description.audio_desc}"

        video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
        if os.path.exists(video_path):
            print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
        else:
            # 基于运镜生成视频
            if self.is_minimax_model:
                # 从 shot_description ff_vis_char_idxs 和 lf_vis_char_idxs 中过滤包含在镜头中的角色。并集 + 唯一
                # vis_char_idxs = []
                # seen = set()
                # for idx in [*shot_description.ff_vis_char_idxs, *shot_description.lf_vis_char_idxs]:
                #     if idx not in seen:
                #         seen.add(idx)
                #         vis_char_idxs.append(idx)
                if workflow_type == "ref":
                    main_characters = [characters[idx] for idx in shot_description.characters]

                    for character in main_characters:
                        if shot_description.characters_path and str(character.idx) in shot_description.characters_path:
                            character_path = os.path.join(self.working_dir, "../character_portraits", f"{character.idx}_{character.identifier_in_scene}/{shot_description.characters_path[str(character.idx)]}.png")
                        else:
                            character_path = os.path.join(self.working_dir, "../character_portraits", f"{character.idx}_{character.identifier_in_scene}/front.png")

                        frame_paths.append(character_path)
                
                video_output = await self.video_generator.generate_single_video(
                    prompt=final_prompt,
                    reference_image_paths=frame_paths,
                    audio_path=dialogue_audio_path,
                    duration=int(shot_description.duration or 5.0),
                    enable_schedule_mode=True,
                    workflow_name=workflow_type,
                )
            else:
                video_output = await self.video_generator.generate_single_video(
                    prompt=final_prompt + "\n整个过程光影流动自然，人物动作与物体形变连贯衔接。",
                    reference_image_paths=frame_paths,
                    audio_path=dialogue_audio_path,
                    duration=int(shot_description.shot_duration or 5.0),
                    use_xianxia_lora=shot_description_with_dialogues.use_xianxia_lora,
                    is_small_people=shot_description_with_dialogues.is_small_people,
                    enable_schedule_mode=True,
                )

            if video_output:
                # 后续统一成 1680
                video_output.save(video_path)

                # 小镜头，提取最后一帧
                last_frame_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png")

                if os.path.exists(video_path):
                    subprocess.run([
                        "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
                        "-update", "1", "-q:v", "1", last_frame_path,
                    ], check=True, capture_output=True)

                print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")
            else:
                if self.comfyui_enable:
                    print(f"[COMFYUI SCHEDULE DETAIL]:: file_path: {video_path}")


    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        # environment: EnvironmentDesign,
    ) -> List[StoryboardShot]:
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [StoryboardShot.model_validate(shot) for shot in storyboard]
            print(f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            print(f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.generate_storyboard(
                script=script,
                characters=characters,
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
        # 限制 LLM 并发避免 429 Too Many Requests
        semaphore = asyncio.Semaphore(3)

        async def _limited(sbd: ShotBriefDescription):
            async with semaphore:
                await asyncio.sleep(0.5)  # 错开请求间隔
                return await self.decompose_visual_description_for_single_shot_brief_description(
                    sbd, characters
                )

        tasks = [_limited(sbd) for sbd in shot_brief_descriptions]
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

        return shot_description


    async def generate_narration_audio(
        self,
        user_requirement: str,
        storyboard: List[StoryboardShot],
    ):
        """
        生成旁白音频。只生成，不合并
        使用 NarrationAgent 生成旁白音频。如果有转场，将转场时长计入对应 shot 的尾部静音，
        保证旁白音频总时长 = 最终视频总时长（转场视频不加旁白）。
        """

        print(f"🎬 Starting generating narration audio...")
        narration_desc_path = os.path.join(self.working_dir, "narration_desc.json")
        
        # 读取或生成旁白描述
        if os.path.exists(narration_desc_path):
            with open(narration_desc_path, 'r', encoding='utf-8') as f:
                narration_data = json.load(f)
            narration_desc = [NarrationShot.model_validate(item) for item in narration_data]
            print(f"🚀 Loaded narration description from existing file.")
        else:
            print(f"🔍 Generating narration description...")

            # 调用 NarrationAgent.generate_narration 生成旁白
            narration_desc = await self.narration_agent.generate_narration(
                storyboard=storyboard,
                user_requirement=user_requirement
            )
            
            # 保存旁白描述
            narration_data = [item.model_dump() for item in narration_desc]
            with open(narration_desc_path, 'w', encoding='utf-8') as f:
                json.dump(narration_data, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Generated narration description and saved to {narration_desc_path}.")
        
        if self.check_interrupt("intro_narration_desc"):
            return

        if not self.comfyui_enable:
            return

        # 4. 生成旁白音频（shot 尾部静音中自然包含转场时长）
        await self.narration_agent.generate_audio(
            narration=narration_desc,
            output_path=self.working_dir,
        )


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
            "-vf", "scale=1690:960:force_original_aspect_ratio=decrease,pad=1690:960:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            video_path,
        ], check=True, capture_output=True)
        os.remove(temp_video_path)
