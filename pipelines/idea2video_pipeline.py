import os
import logging
import subprocess
from agents import Screenwriter, CharacterExtractor, CharacterPortraitsGenerator, VoiceDesigner
from agents.transition_agent import TransitionDirector, TransFragment
from pipelines.script2video_pipeline import Script2VideoPipeline
from pipelines.script2video_pipeline_v2 import Script2VideoPipelineV2
from pipelines.hanzi_pipeline import HanziPipeline
from interfaces import CharacterInScene, ShotDescription
from typing import List, Dict, Optional, Tuple
import asyncio
import json
import aiohttp
from moviepy import VideoFileClip, concatenate_videoclips
import yaml
from tools.render_backend import RenderBackend
from utils.provider_presets import create_chat_model
from utils.voice import register_voice, VoiceManager
from interfaces import *

class Idea2VideoPipeline:
    def __init__(
        self,
        chat_model: str,
        image_generator: str,
        video_generator: str,
        audio_generator: str,
        working_dir: str,
        comfyui_base_url: str,
        interrupt_step: str = None,
        mode: str = "normal",
        hanzi: str = None,
        new_character: List[str] = None,
        relate_hanzi: List[str] = None,
        gacha_config: dict = None,
        config: dict = None,
        minor_video_generator: str = None,
    ):
        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.minor_video_generator = minor_video_generator
        self.audio_generator = audio_generator
        self.working_dir = working_dir
        self.comfyui_base_url = comfyui_base_url
        self.interrupt_step = interrupt_step
        self.mode = mode
        self.hanzi = hanzi
        self.new_character = new_character
        self.relate_hanzi = relate_hanzi
        self.gacha_config = gacha_config
        self.comfyui_enable = False
        self.config = config
        os.makedirs(self.working_dir, exist_ok=True)

        self.screenwriter = Screenwriter(chat_model=self.chat_model)
        self.character_extractor = CharacterExtractor(
            chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(
            image_generator=self.image_generator)
        self.transition_director = TransitionDirector(chat_model=self.chat_model)
        self.voice_designer = VoiceDesigner(chat_model=self.chat_model)

        os.makedirs(os.path.join(self.working_dir, "workflows"), exist_ok=True)
        os.makedirs(os.path.join(self.working_dir, "transitions"), exist_ok=True)
        os.makedirs(os.path.join(self.working_dir, "voices"), exist_ok=True)

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
            audio_generator=backend.audio_generator,
            working_dir=config["working_dir"],
            comfyui_base_url=config["comfyui_base_url"],
            interrupt_step=config["interrupt_step"],
            mode=config["mode"],
            hanzi=config.get("hanzi", ""),
            new_character=config.get("new_character", "").split(",") if config.get("new_character") else None,
            relate_hanzi=config.get("relate_hanzi", "").split(",") if config.get("relate_hanzi") else None,
            gacha_config=config["gacha_config"],
            config=config,
            minor_video_generator=backend.minor_video_generator,
        )

    async def _check_comfyui_health(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.comfyui_base_url}/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def extract_characters(
        self,
        story: str,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")
        need_extra = True
        characters = None
        lack_characters = []

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(
                character) for character in characters]
            need_extra = False
            
            # 检查新角色是否存在，不存在也要创建
            if self.new_character:
                exist_characters = [character.identifier_in_scene for character in characters]

                for character in self.new_character:
                    if character not in exist_characters:
                        lack_characters.append(character)
                        need_extra = True

        if need_extra:
            characters_response = await self.character_extractor.extract_characters(story)
            
            if characters and lack_characters:
                # 新增角色
                new_idx = len(characters)
                
                for character in characters_response:
                    if character.identifier_in_scene in lack_characters:
                        character.idx = new_idx
                        characters.append(character)
                        new_idx += 1
            else:
                characters = characters_response
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump()
                          for character in characters], f, ensure_ascii=False, indent=4)
            print(
                f"✅ Extracted {len(characters)} characters from story and saved to {save_path}.")
        else:
            print(f"🚀 Loaded {len(characters)} characters from existing file.")

        voice_list = VoiceManager.instance().list_all()

        # 检查角色是否存在音色，不存在则创建音色
        # for character in characters:
        #     if character.identifier_in_scene not in voice_list:
        #         # 如果 identifier_in_scene 包含 “字” 字符，则跳过
        #         if "字" in character.identifier_in_scene and len(character.identifier_in_scene) == 2:
        #             continue

        #         if self.comfyui_enable:
        #             voice = await self.voice_designer.design_and_register(character, self.audio_generator)
        #             register_voice(name=voice["character"], gender=voice["gender"])

        return characters

    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
    ):
        character_portraits_registry_path = os.path.join(
            self.working_dir, "character_portraits_registry.json")
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}

        # 为每个角色生成肖像图片
        tasks = [
            self.generate_portraits_for_single_character(character, style)
            for character in characters
            if character.identifier_in_scene not in character_portraits_registry
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)

                if self.comfyui_enable:
                    with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                        json.dump(character_portraits_registry,
                                f, ensure_ascii=False, indent=4)

            print(
                f"✅ Completed character portrait generation for {len(characters)} characters.")
        else:
            print(
                "🚀 All characters already have portraits, skipping portrait generation.")

        return character_portraits_registry

    async def develop_story(
        self,
        idea: str,
        user_requirement: str,
        hanzi_story: Optional[str] = None,
    ):
        save_path = os.path.join(self.working_dir, "story.txt")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                story = f.read()
            print(f"🚀 Loaded story from existing file.")
        else:
            print("🧠 Developing story...")
            story = await self.screenwriter.develop_story(idea=idea, user_requirement=user_requirement, hanzi_story=hanzi_story)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(story)
            print(f"✅ Developed story and saved to {save_path}.")

        return story

    async def write_script_based_on_story(
        self,
        story: str,
        user_requirement: str,
    ):
        save_path = os.path.join(self.working_dir, "script.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                script = json.load(f)
            print(f"🚀 Loaded script from existing file.")
        else:
            print("🧠 Writing script based on story...")
            script = await self.screenwriter.write_script_based_on_story(story=story, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=4)
            print(f"✅ Written script based on story and saved to {save_path}.")
        return script

    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
    ):
        character_dir = os.path.join(
            self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            if self.comfyui_enable:
                front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
                front_portrait_output.save(front_portrait_path)

        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path):
            pass
        else:
            if self.comfyui_enable:
                side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
                side_portrait_output.save(side_portrait_path)

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path):
            pass
        else:
            if self.comfyui_enable:
                back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
                back_portrait_output.save(back_portrait_path)

        print(
            f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")

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

    async def generate_scene_transitions(self, scene_scripts: List[str], style: str) -> List[Optional[Tuple[str, float]]]:
        """
        生成相邻场景间的转场视频，latent continuity transition

        对每对相邻场景 (scene_i, scene_{i+1})：
        1. 获取前一个场景的最后一个镜头 (尾帧 + 描述)
        2. 获取后一个场景的第一个镜头 (首帧 + 描述)
        3. 调用 TransitionDirector 生成桥接转场描述
        4. 生成转场视频并保存
        """

        if len(scene_scripts) < 2:
            return []

        transition_dir = os.path.join(self.working_dir, "transitions")
        os.makedirs(transition_dir, exist_ok=True)

        # ── 转场描述缓存 ──
        trans_desc_cache_path = os.path.join(transition_dir, "scene_trans_descriptions.json")
        transition_fragments = []
        if os.path.exists(trans_desc_cache_path):
            with open(trans_desc_cache_path, "r", encoding="utf-8") as f:
                transition_data = json.load(f)
            transition_fragments = [TransFragment.model_validate(item) for item in transition_data]

        num_transitions = len(scene_scripts) - 1
        transition_results: List[Optional[Tuple[str, float]]] = []

        for i in range(num_transitions):
            prev_scene_dir = os.path.join(self.working_dir, f"scene_{i}")
            next_scene_dir = os.path.join(self.working_dir, f"scene_{i+1}")

            # ── 转场视频路径 ──
            transition_video_path = os.path.join(
                transition_dir, f"transition_scene_{i}_to_{i+1}.mp4"
            )

            if os.path.exists(transition_video_path):
                print(f"🚀 Skipped scene transition {i}->{i+1}, video already exists.")
                clip = VideoFileClip(transition_video_path)
                transition_results.append((transition_video_path, clip.duration))
                clip.close()
                continue

            if not self.comfyui_enable:
                transition_results.append(None)
                continue

            # ── 1. 获取前一个场景的最后一个镜头 ──
            prev_last_shot = self._load_last_shot(prev_scene_dir)
            if prev_last_shot is None:
                print(f"⚠️ Cannot find last shot in scene {i}, skipping transition {i}->{i+1}.")
                transition_results.append(None)
                continue

            prev_last_frame_path = os.path.join(
                prev_scene_dir, "shots", str(prev_last_shot.idx), "last_frame.png"
            )
            prev_last_frame_path = self._ensure_frame(
                prev_last_frame_path,
                os.path.join(prev_scene_dir, "shots", str(prev_last_shot.idx), "video.mp4"),
                label=f"scene {i} last frame",
            )
            if prev_last_frame_path is None:
                transition_results.append(None)
                continue

            # ── 2. 获取后一个场景的第一个镜头 ──
            next_first_shot = self._load_shot_by_idx(next_scene_dir, 0)
            if next_first_shot is None:
                print(f"⚠️ Cannot find first shot (idx=0) in scene {i+1}, skipping transition {i}->{i+1}.")
                transition_results.append(None)
                continue

            next_first_frame_path = os.path.join(
                next_scene_dir, "shots", "0", "first_frame.png"
            )
            if not os.path.exists(next_first_frame_path):
                print(f"⚠️ No first frame found for scene {i+1} shot 0, skipping transition {i}->{i+1}.")
                transition_results.append(None)
                continue

            print(f"🎬 Generating scene transition {i} -> {i+1}...")

            # ── 3. 生成 / 复用转场描述 ──
            trans_fragment = next(filter(lambda x: x.idx == i, transition_fragments), None)

            if trans_fragment is None:
                trans_fragment = await self.transition_director.design_transition(
                    previous_shot=prev_last_shot.visual_desc,
                    next_shot=next_first_shot.visual_desc,
                    previous_last_frame=prev_last_shot.lf_desc if prev_last_shot.lf_desc else "",
                    next_first_frame=next_first_shot.ff_desc if next_first_shot.ff_desc else "",
                    environment_style=style,
                )
                trans_fragment.idx = i
                transition_fragments.append(trans_fragment)

            # ── 4. 生成转场视频 ──
            trans_output = await self.video_generator.generate_single_video(
                prompt=trans_fragment.motion_desc,
                reference_image_paths=[prev_last_frame_path, next_first_frame_path],
                duration=trans_fragment.duration,
            )
            trans_output.save(transition_video_path)
            # self._scale_video(trans_output, transition_video_path)

            # ── 5. 记录结果 ──
            clip = VideoFileClip(transition_video_path)
            transition_results.append((transition_video_path, clip.duration))
            clip.close()

            print(f"☑️ Generated scene transition {i} -> {i+1}, saved to {transition_video_path}.")

        # ── 回写转场描述缓存 ──
        if len(transition_fragments) > 0:
            with open(trans_desc_cache_path, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in transition_fragments], f, ensure_ascii=False, indent=4)
            print(f"💾 Saved {len(transition_fragments)} scene transition descriptions to {trans_desc_cache_path}.")

        transition_data = []

        for trans in transition_fragments:
            if trans.enable:
                transition_data.append(transition_results[trans.idx])
            else:
                transition_data.append(None)

        return transition_data

    # ── 辅助方法 ──

    @staticmethod
    def _load_storyboard(scene_dir: str) -> Optional[List]:
        """加载场景的 storyboard"""
        storyboard_path = os.path.join(scene_dir, "storyboard.json")
        if not os.path.exists(storyboard_path):
            print(f"⚠️ storyboard.json not found in {scene_dir}")
            return None
        with open(storyboard_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_shot_by_idx(scene_dir: str, shot_idx: int) -> Optional[ShotDescription]:
        """加载场景中指定索引的镜头描述"""
        shot_desc_path = os.path.join(scene_dir, "shots", str(shot_idx), "shot_description.json")
        if not os.path.exists(shot_desc_path):
            print(f"⚠️ Shot description not found: {shot_desc_path}")
            return None
        with open(shot_desc_path, "r", encoding="utf-8") as f:
            return ShotDescription.model_validate(json.load(f))

    @classmethod
    def _load_last_shot(cls, scene_dir: str) -> Optional[ShotDescription]:
        """加载场景的最后一个镜头"""
        storyboard = cls._load_storyboard(scene_dir)
        if not storyboard:
            return None
        last_idx = len(storyboard) - 1
        return cls._load_shot_by_idx(scene_dir, last_idx)

    @staticmethod
    def _ensure_frame(frame_path: str, video_path: str, label: str = "frame") -> Optional[str]:
        """确保帧文件存在，必要时从视频中提取"""
        if os.path.exists(frame_path):
            return frame_path
        if os.path.exists(video_path):
            print(f"📸 Extracting {label} from video...")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
                    "-update", "1", "-q:v", "1", frame_path,
                ], check=True, capture_output=True)
                print(f"☑️ Extracted {label}.")
                return frame_path
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to extract {label}: {e}")
                return None
        print(f"⚠️ No video found for {label}, skipping.")
        return None

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


    def check_interrupt(self, step_name: str):
        """检查是否中断"""
        
        return True if self.interrupt_step ==  step_name else False

    async def __call__(
        self,
        idea: str,
        user_requirement: str,
        style: str,
    ):
        # 检查 comfyui 的状态
        self.comfyui_enable = await self._check_comfyui_health()

        hanzi_idea = ""
        # 汉字模式
        if self.mode == "hanzi":
            # 汉字模式
            hanzi_working_dir = os.path.join(self.working_dir, "hanzi")
            hanzi_pipeline = HanziPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                audio_generator=self.audio_generator,
                minor_video_generator=self.minor_video_generator,
                working_dir=hanzi_working_dir,
                hanzi=self.hanzi,
                relate_hanzi=self.relate_hanzi,
                interrupt_step=self.interrupt_step,
                comfyui_enable=self.comfyui_enable,
                disable_transition=self.config.get("disable_transition", False)
            )
            hanzi_video_path = await hanzi_pipeline()
            
            # if not hanzi_video_path or not os.path.exists(hanzi_video_path):
            #     print(f"❌ Hanzi pipeline failed to generate video.")
            #     return
            hanzi_idea_path = os.path.join(hanzi_working_dir, "idea.txt")

            if os.path.exists(hanzi_idea_path):
                with open(hanzi_idea_path, "r", encoding="utf-8") as f:
                    hanzi_idea = f.read()

        if self.interrupt_step in ["crawl", "download", "idea", "transition", "video", "hanzi"]:
            return

        # 生成故事
        # 去掉首尾空格
        idea = idea.strip()
        if idea == "汉字教学":
            story = await self.develop_story(idea=hanzi_idea, user_requirement=user_requirement)
        else:   
            story = await self.develop_story(idea=idea, user_requirement=user_requirement, hanzi_story=hanzi_idea)
        
        if self.check_interrupt("story"):
            return

        # 生成故事的所有角色特征
        characters = await self.extract_characters(story=story)
        
        if self.check_interrupt("character"):
            return

        # 生成角色肖像（正面、侧面、背面），并保存路径
        character_portraits_registry = await self.generate_character_portraits(
            characters=characters,
            character_portraits_registry=None,
            style=style,
        )
        # character_portraits_registry = {}
        
        if self.check_interrupt("portrait"):
            return

        # 生成场景剧本
        scene_scripts = await self.write_script_based_on_story(story=story, user_requirement=user_requirement)
        
        if self.check_interrupt("script"):
            return

        all_video_paths = []

        if self.interrupt_step and self.interrupt_step.startswith("intro_"):
            intro_working_dir = os.path.join(self.working_dir, "intro")
            intro_script2video_pipeline = Script2VideoPipelineV2(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                audio_generator=self.audio_generator,
                working_dir=intro_working_dir,
                interrupt_step=self.interrupt_step,
                comfyui_enable=self.comfyui_enable,
            )
            # 取 working_dir/intro.md 作为脚本
            intro_script_path = os.path.join(intro_working_dir, "intro.md")
            with open(intro_script_path, "r", encoding="utf-8") as f:
                intro_script = f.read()

            intro_final_video_path = await intro_script2video_pipeline(
                script=intro_script,
                user_requirement=user_requirement,
                style=style,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
            )
            return

        if self.mode == "gacha":
            # 抽卡模式。只需要处理对应场景
            scene_idx = self.gacha_config["scene"]
            scene_script = scene_scripts[scene_idx]
            scene_working_dir = os.path.join(self.working_dir, f"scene_{scene_idx}")
            print(f"🎬 Gacha Mode:: Starting scene {scene_idx} video generation...")
            script2video_pipeline = Script2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                audio_generator=self.audio_generator,
                working_dir=scene_working_dir,
                interrupt_step=self.interrupt_step,
                gacha_config=self.gacha_config,
            )
            await script2video_pipeline(
                script=scene_script,
                user_requirement=user_requirement,
                style=style,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
            )
        else:
            # 生成场景视频，拆解成镜头再合成
            for idx, scene_script in enumerate(scene_scripts):
                # if idx == 1:
                #     continue

                scene_working_dir = os.path.join(self.working_dir, f"scene_{idx}")
                os.makedirs(scene_working_dir, exist_ok=True)
                print(f"🎬 Starting scene {idx} video generation...")
                script2video_pipeline = Script2VideoPipeline(
                    chat_model=self.chat_model,
                    image_generator=self.image_generator,
                    video_generator=self.video_generator,
                    audio_generator=self.audio_generator,
                    working_dir=scene_working_dir,
                    interrupt_step=self.interrupt_step,
                    comfyui_enable=self.comfyui_enable,
                )
                final_video_path = await script2video_pipeline(
                    script=scene_script,
                    user_requirement=user_requirement,
                    style=style,
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                )
                all_video_paths.append(final_video_path)
                print(f"☑️ Completed scene {idx} video generation, saved to {final_video_path}.")

            if self.interrupt_step is not None and self.interrupt_step not in ["scene_transition", "final"]:
                return

            # latent continuity transition
            # scene_transitions = await self.generate_scene_transitions(
            #     scene_scripts=scene_scripts,
            #     style=style,
            # )
            scene_transitions = []
            
            if self.check_interrupt("scene_transition"):
                return

            # 合并所有场景视频
            final_video_path = os.path.join(self.working_dir, "final_video.mp4")
            if os.path.exists(final_video_path):
                print(f"🚀 Skipped concatenating videos, already exists.")
            else:
                print(f"🎬 Starting concatenating videos...")
                video_clips = []
                for idx, scene_video_path in enumerate(all_video_paths):
                    video_clips.append(VideoFileClip(scene_video_path, audio=True))

                    if idx < len(scene_transitions) and scene_transitions[idx] is not None:
                        trans_path, _ = scene_transitions[idx]
                        video_clips.append(VideoFileClip(trans_path, audio=False))

                intro_video_path = os.path.join(self.working_dir, "intro", "final_video.mp4")
                if os.path.exists(intro_video_path):
                    video_clips.insert(0, VideoFileClip(intro_video_path, audio=True))
                
                # if self.mode == "hanzi" and hanzi_video_path:
                #     video_clips.append(VideoFileClip(hanzi_video_path, audio=True))
                
                final_video = concatenate_videoclips(video_clips, method="compose")
                final_video.write_videofile(final_video_path, codec="libx264", preset="medium", audio_codec="aac", fps=None, audio_bitrate="192k")
                print(f"☑️ Concatenated videos, saved to {final_video_path}.")
            return final_video_path
