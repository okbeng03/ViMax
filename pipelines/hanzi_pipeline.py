"""
HanziPipeline: 汉字演变视频生成流水线

功能：
1. 从字典网站爬取汉字的基础解释和字源字形
2. 将 SVG 图片转换为 PNG
3. 生成汉字演变教程的创意
4. 使用 agents/hanzi_evolution.py 生成演变过渡动画
5. 保存汉字字形到 characters.json 和 character_portraits_registry.json
"""

import os
import json
import logging
import re
import yaml
import shutil
import cairosvg
from typing import Any, List
from PIL import Image
from tools.render_backend import RenderBackend
from langchain.chat_models import init_chat_model
from moviepy import VideoFileClip, concatenate_videoclips

import requests
from bs4 import BeautifulSoup
from langchain.chat_models.base import BaseChatModel

from utils.image import download_image
from utils.provider_presets import resolve_chat_model_config
from agents.hanzi_creative_agent import HanziCreativeAgent
from agents.hanzi_evolution_agent import HanziEvolutionAgent, EvolutionTransitions

logger = logging.getLogger(__name__)

glyph_type_descriptions = {
    "甲骨文": "如果未明确，泛指古人的汉字",
    "楷书": "如果未明确，泛指现在的汉字"
}

class HanziPipeline:
    def __init__(
        self,
        chat_model: BaseChatModel,
        image_generator: Any,
        video_generator: Any,
        working_dir: str,
        hanzi: str,
        relate_hanzi: List[str] | None = None,
        interrupt_step: str | None = None,
    ):
        self.chat_model: BaseChatModel = chat_model
        self.image_generator: Any = image_generator
        self.video_generator: Any = video_generator
        self.working_dir: str = working_dir
        self.hanzi: str = hanzi
        self.relate_hanzi: List[str] | None = relate_hanzi
        self.interrupt_step: str | None = interrupt_step
        
        os.makedirs(self.working_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.working_dir, "character_portraits"), exist_ok=True)
        
        # 初始化 agents
        self.creative_agent: HanziCreativeAgent = HanziCreativeAgent(chat_model=chat_model)
        self.evolution_agent: HanziEvolutionAgent = HanziEvolutionAgent(chat_model=chat_model)

    @classmethod
    def init_from_config(cls, config_path: str):
        """从配置文件初始化"""
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        chat_model_args = resolve_chat_model_config(config["chat_model"]["init_args"])
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(config)
        hanzi = config.get("hanzi")

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config.get("working_dir", f"./output/hanzi_{hanzi}"),
            hanzi=hanzi,
            interrupt_step=config.get("interrupt_step"),
        )

    def check_interrupt(self, step_name: str) -> bool:
        """检查是否中断"""
        return self.interrupt_step == step_name

    async def crawl_hanzi_info(self, hanzi: str | None = None, working_dir: str | None = None) -> dict[str, Any]:
        """
        从字典网站爬取汉字信息
        
        Args:
            hanzi: 要爬取的汉字，默认使用 self.hanzi
        
        Returns:
            包含基础解释、字源字形的字典
        """
        
        if not hanzi:
            hanzi = self.hanzi
            
        if not working_dir:
            working_dir = self.working_dir
            
        save_path = os.path.join(working_dir, "hanzi_info.json")
        
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                hanzi_info = json.load(f)
            print(f"🚀 Loaded hanzi info from existing file.")
            return hanzi_info

        print(f"🕷️ Crawling hanzi info for '{hanzi}' from zdic.net...")
        
        url = f"https://www.zdic.net/hans/{hanzi}"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取基础解释
            basic_definitions = self._extract_basic_explanation(soup)
            
            # 提取字源字形
            glyphs = self._extract_glyphs(soup)
            
            hanzi_info = {
                "hanzi": hanzi,
                "basic_definitions": basic_definitions,
                "glyphs": glyphs,
                "source_url": url,
            }
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(hanzi_info, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Crawled {hanzi} info and saved to {save_path}")
            return hanzi_info
            
        except Exception as e:
            logger.error(f"Failed to crawl hanzi info: {e}")
            raise

    def _extract_basic_explanation(self, soup: BeautifulSoup) -> str:
        """提取基础解释"""

        definitions = []
        
        # 用户提供的 HTML 结构: div[data-type-block="基本解释"] > div.definitions > ol > li
        basic_block = soup.find('div', attrs={'data-type-block': '基本解释'})
        if basic_block:
            definitions_div = basic_block.find('div', class_='definitions')
            if definitions_div:
                pinyins = basic_block.select('p > span.dicpy')
                ols = definitions_div.select('ol')
                for idx, ol in enumerate(ols):
                    pinyin = pinyins[idx].contents[0].strip()
                    explanations = []
                    for li in ol.find_all('li'):
                        text = li.get_text(strip=True)
                        if text:
                            explanations.append(text)
                            
                    definitions.append({
                        "pinyin": pinyin,
                        "explanations": explanations,
                    })
        
        return definitions

    def _extract_glyphs(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """
        提取字源字形（甲骨文、金文、楚系簡帛、楷書）
        
        HTML结构:
        div[data-type-block="字源字形"] > div.definitions > table.zyyb
        
        Returns:
            字形列表，每个字形包含 type 和 svg_url
        """
        glyphs = []
        
        # 用户需要的字形类型映射
        target_types = ['甲骨文', '金文', '楚系简帛', '楷书']
        
        # 查找字源字形区域
        glyph_block = soup.find('div', attrs={'data-type-block': '字源字形'})
        if not glyph_block:
            return glyphs
        
        table = glyph_block.find('table', class_='zyyb')
        if not table:
            return glyphs
        
        # tbody = table.find('tbody')
        # if not tbody:
        #     return glyphs
        
        rows = table.find_all('tr')
        if len(rows) < 2:
            return glyphs
        
        # 第一行是表头，建立列索引到类型的映射
        header_row = rows[0]
        header_cells = header_row.find_all(['th', 'td'])
        col_type_map = {}  # col_index -> type_name
        
        for idx, cell in enumerate(header_cells):
            # 首先要求 cell 非 display: none
            style = cell.get('style', '').lower()
            if 'display: none' in style or 'display:none' in style:
                continue

            cell_text = cell.get_text(strip=True)
            
            # 标准化类型名称
            if cell_text in ['甲骨文']:
                col_type_map[idx] = '甲骨文'
            elif cell_text in ['金文']:
                col_type_map[idx] = '金文'
            elif cell_text in ['楚系簡帛', '楚系簡帛', '楚簡', '楚簡文字']:
                col_type_map[idx] = '楚系简帛'
            elif cell_text in ['楷書', '楷书']:
                col_type_map[idx] = '楷书'
        
        # 遍历数据行，提取图片
        for row in [rows[1]]:
            cells = row.find_all('td')
            for col_idx, cell in enumerate(cells):
                if col_idx not in col_type_map:
                    continue
                
                glyph_type = col_type_map[col_idx]
                if glyph_type not in target_types:
                    continue
                
                # 查找图片或链接
                img = cell.find('img', attrs={'data-original': re.compile(r'\.svg', re.IGNORECASE)})
                if img:
                    # 优先使用 data-original，否则用 src
                    svg_url = img.get('data-original') or img.get('src', '')
                    
                    if svg_url:
                        # 转换为完整 URL
                        if svg_url.startswith('//'):
                            svg_url = 'https:' + svg_url
                        elif svg_url.startswith('/'):
                            svg_url = 'https://www.zdic.net' + svg_url
                        
                        # 查找说明文字
                        spans = cell.find_all('span')
                        alt = ' '.join([s.get_text(strip=True) for s in spans]) if spans else ''
                        
                        glyphs.append({
                            "type": glyph_type,
                            "svg_url": svg_url,
                            "alt": alt,
                        })
                        
                        # 已找到，col_type_map 要移除对应 idx
                        del col_type_map[col_idx]
        
        return glyphs

    async def download_and_convert_svg_to_png(
        self, 
        hanzi_info: dict[str, Any],
        working_dir: str | None = None,
        target_types: list[str] | None = None,
    ) -> dict[str, str]:
        """
        下载 SVG 图片并转换为 PNG
        
        Args:
            hanzi_info: 汉字信息字典
            working_dir: 工作目录，默认使用 self.working_dir
            target_types: 目标字形类型列表，默认下载所有类型
        
        Returns:
            字形类型到 PNG 路径的映射
        """
        target_dir = working_dir or self.working_dir
        glyphs = hanzi_info.get("glyphs", [])
        glyph_png_paths = {}
        
        for glyph in glyphs:
            glyph_type = glyph["type"]
            
            # 如果指定了目标类型，只处理目标类型
            if target_types and glyph_type not in target_types:
                continue
            
            svg_url = glyph["svg_url"]
            svg_path = os.path.join(target_dir, f"{glyph_type}.svg")
            png_path = os.path.join(target_dir, f"{glyph_type}.png")
            
            # 如果 PNG 已存在，跳过
            if os.path.exists(png_path):
                glyph_png_paths[glyph_type] = png_path
                print(f"🚀 PNG already exists for {glyph_type}")
                continue
            
            # 下载 SVG
            if not os.path.exists(svg_path):
                try:
                    download_image(svg_url, svg_path)
                    print(f"✅ Downloaded {glyph_type} SVG from {svg_url}")
                except Exception as e:
                    logger.warning(f"Failed to download SVG for {glyph_type}: {e}")
                    continue
            
            # SVG 转 PNG
            try:
                self._convert_svg_to_png(svg_path, png_path)
                glyph_png_paths[glyph_type] = png_path
                print(f"✅ Converted {glyph_type} SVG to PNG")
            except Exception as e:
                logger.warning(f"Failed to convert SVG to PNG for {glyph_type}: {e}")
        
        return glyph_png_paths

    def _convert_svg_to_png(self, svg_path: str, png_path: str, size: int = 512):
        """
        将 SVG 文件转换为 PNG，同时将透明底黑色字转换为白底黑色字
        
        Args:
            svg_path: SVG 文件路径
            png_path: 输出 PNG 文件路径
            size: 输出图片尺寸
        """
        import subprocess
        
        # 临时文件路径（用于 cairosvg 转换）
        temp_png_path = png_path + ".temp.png"
        
        try:
            # 使用 cairosvg 转换
            cairosvg.svg2png(
                url=svg_path,
                write_to=temp_png_path,
                output_width=size,
                output_height=size
            )
        except ImportError:
            logger.warning("cairosvg not installed, trying alternative method...")
            # 备选方案：使用 inkscape 或 rsvg-convert
            try:
                subprocess.run([
                    'rsvg-convert', '-w', str(size), '-h', str(size),
                    '-o', temp_png_path, svg_path
                ], check=True)
            except (OSError, subprocess.CalledProcessError):
                # 最后备选：使用 PIL 读取（可能不支持某些 SVG）
                logger.error("No SVG converter available. Please install cairosvg: pip install cairosvg")
                raise
        
        # 将透明底黑色字转换为白底黑色字
        self._convert_transparent_black_to_white_bg(temp_png_path, png_path)
        
        # 清理临时文件
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)
    
    def _convert_transparent_black_to_white_bg(self, input_path: str, output_path: str):
        """
        将透明底黑色字转换为白底黑色字
        
        Args:
            input_path: 输入图片路径（RGBA，可能有透明背景和黑色文字）
            output_path: 输出图片路径（RGB，白底黑字）
        """
        # 打开图片并转换为 RGBA
        img = Image.open(input_path).convert("RGBA")
        
        # 创建白色背景
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        
        # 将原图作为蒙版粘贴到白色背景上（保留黑色文字区域）
        white_bg.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为蒙版
        
        # 保存为 PNG
        white_bg.save(output_path, "PNG")
        print(f"✅ Converted to white background: {output_path}")

    async def generate_creative_idea(self, hanzi_info: dict[str, Any]) -> str:
        """
        根据汉字解释和字源字形生成创意
        - 甲骨文象形
        - 演变动画
        - 基础解释 - 多音字
        - 有趣的事
        - 字笔画
        
        Returns:
            创意描述（包含有趣知识点）
        """
        save_path = os.path.join(self.working_dir, "idea.txt")
        
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                idea = f.read()
            print(f"🚀 Loaded creative idea from existing file.")
            return idea

        print(f"🧠 Generating creative idea for '{self.hanzi}'...")
        
        basic_definitions = hanzi_info.get("basic_definitions", "")
        glyphs = hanzi_info.get("glyphs", [])
        glyph_types = [g["type"] for g in glyphs]
        
        # 使用 creative agent 生成创意（包含有趣知识点，由 LLM 生成）
        idea_story = await self.creative_agent.generate_creative(
            hanzi=self.hanzi,
            basic_definitions=basic_definitions,
            glyph_types=glyph_types,
        )
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(idea_story)
        
        print(f"✅ Generated creative idea and saved to {save_path}")
        return idea_story

    async def generate_transitions(
        self, 
        glyph_png_paths: dict[str, str],
    ) -> list[str]:
        """
        使用 hanzi_evolution agent 生成字形过渡描述
        
        Args:
            glyph_png_paths: 字形 PNG 路径字典
        
        Returns:
            过渡描述列表
        """
        
        print(f"🎬 Generating transition descriptions for {len(glyph_png_paths)} glyphs...")
        evolution_path = os.path.join(self.working_dir, "transition.json")
        transitions = []
        
         # 定义字形演变顺序
        glyph_order = ['甲骨文', '金文', '楚系简帛', '楷书']
        
        # 获取按顺序排列的字形路径
        ordered_glyphs = []
        for glyph_type in glyph_order:
            if glyph_type in glyph_png_paths:
                ordered_glyphs.append({
                    "type": glyph_type,
                    "path": glyph_png_paths[glyph_type]
                })
                
                
        if os.path.exists(evolution_path):
            with open(evolution_path, "r", encoding="utf-8") as f:
                transitions = json.load(f)

            transitions = EvolutionTransitions.model_validate(transitions)
            print(f"🚀 Transition description already exists: {evolution_path}")
        else:
            # 使用 evolution agent 生成过渡描述
            transitions = await self.evolution_agent.generate_transition_description(
                hanzi=self.hanzi,
                glyphs=ordered_glyphs,
                retry_timeout=300,
            )
        
            with open(evolution_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(transitions.model_dump(), ensure_ascii=False, indent=4))
                print(f"✅ Generated transition description: {evolution_path}")
                
        return transitions

    async def generate_evolution_videos(
        self, 
        glyph_png_paths: dict[str, str],
        transitions: EvolutionTransitions,
    ) -> list[str]:
        """
        使用 hanzi_evolution agent 生成演变过渡动画
        
        Args:
            glyph_png_paths: 字形 PNG 路径字典
            transitions: 过渡描述列表
        
        Returns:
            生成的所有视频路径列表
        """

        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        evolution_dir = os.path.join(self.working_dir, "evolution")
        video_paths = []
        
        if len(glyph_png_paths) < 2:
            print("⚠️ Not enough glyphs to generate evolution video")
            return video_paths
        
        print(f"🎬 Generating evolution videos for {len(glyph_png_paths)} glyphs...")
        
        # 生成相邻字形之间的过渡视频
        for idx, transition in enumerate(transitions.indices):
            from_type = transition.from_type
            to_type = transition.to_type
            from_hanzi_path = glyph_png_paths[from_type]
            to_hanzi_path = glyph_png_paths[to_type]
            
            # 合成带背景的图片路径
            from_composite_path = os.path.join(evolution_dir, f"{from_type}_composite.png")
            to_composite_path = os.path.join(evolution_dir, f"{to_type}_composite.png")
            composite_prompt = "生成一个高质量的合成图片：将Image 0的黑色汉字颜色换成白色粉笔字放在左侧米字格中，书写工整，上下垂直居中。\n黑板背景、左侧米字格子、整体构图、右侧文字参考 Image 1。右下角的小字去掉。\n左侧米字格子中的汉字参考 Image 0，换色白色粉笔颜色。"
            
            # 如果合成图片已存在，跳过生成
            if not os.path.exists(from_composite_path):
                bg_path = os.path.join(assets_dir, f"{from_type}.png")
                from_image = await self.image_generator.generate_single_image(
                    prompt=composite_prompt,
                    reference_image_paths=[from_hanzi_path, bg_path],
                    size="2560x1440",
                )
                from_image.save(from_composite_path)
                print(f"✅ Generated composite image for {from_type}")
            
            if not os.path.exists(to_composite_path):
                bg_path = os.path.join(assets_dir, f"{from_type}.png")
                to_image = await self.image_generator.generate_single_image(
                    prompt=composite_prompt,
                    reference_image_paths=[to_hanzi_path, bg_path],
                    size="2560x1440",
                )
                to_image.save(to_composite_path)
                print(f"✅ Generated composite image for {to_type}")
            
            # 视频文件名：甲骨文_to_金文.mp4
            video_filename = f"{idx}_{from_type}_to_{to_type}.mp4"
            transition_video_path = os.path.join(evolution_dir, video_filename)
            
            # 如果视频已存在，跳过
            if os.path.exists(transition_video_path):
                print(f"🚀 Transition video already exists: {video_filename}")
                video_paths.append(transition_video_path)
                continue
            
            try:
                # 生成过渡视频（5秒纯动画，不包含停顿）
                prompt = f"镜头保持不变，开头1s保持首帧静止，1s后左侧米字格子的文字开始按”{transition.description}“变化，同时右侧文字逐渐消失，并过渡到尾帧的右侧文字。整个变化过程持续到4s，然后保持静止。"
                video_output = await self.video_generator.generate_single_video(
                    prompt=prompt,
                    reference_image_paths=[from_composite_path, to_composite_path],
                    duration=5,
                )
                
                # 保存原始视频
                # raw_video_path = os.path.join(evolution_dir, f"{from_type}_to_{to_type}_raw.mp4")
                video_output.save(transition_video_path)
                
                # 通过后期处理添加前后各1秒的停顿（总时长7秒：1s停顿 + 5s动画 + 1s停顿）
                # self._add_hold_frames(raw_video_path, transition_video_path, hold_first=1, hold_last=1)
                video_paths.append(transition_video_path)
                print(f"✅ Generated transition video {from_type} to {to_type}, saved to {transition_video_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate transition video {from_type} to {to_type}: {e}")
        
        return video_paths
    
    def _add_hold_frames(self, input_video_path: str, output_video_path: str, hold_first: float = 1, hold_last: float = 1):
        """
        在视频前后添加停顿帧
        
        Args:
            input_video_path: 输入视频路径
            output_video_path: 输出视频路径
            hold_first: 开头停顿时长（秒）
            hold_last: 结尾停顿时长（秒）
        """
        from moviepy.video.fx.all import freeze
        from moviepy import VideoFileClip, concatenate_videoclips, ImageSequenceClip
        
        try:
            clip = VideoFileClip(input_video_path)
            
            # 获取第一帧和最后一帧
            first_frame = clip.get_frame(0)
            last_frame = clip.get_frame(clip.duration - 0.01)  # 避免边界问题
            
            # 创建停顿片段
            clips = []
            
            # 开头停顿
            if hold_first > 0:
                first_hold = ImageSequenceClip([first_frame], fpss=[clip.fps])
                first_hold = first_hold.with_duration(hold_first)
                clips.append(first_hold)
            
            # 中间原始视频
            clips.append(clip)
            
            # 结尾停顿
            if hold_last > 0:
                last_hold = ImageSequenceClip([last_frame], fpss=[clip.fps])
                last_hold = last_hold.with_duration(hold_last)
                clips.append(last_hold)
            
            # 合并所有片段
            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip.write_videofile(
                output_video_path,
                codec="libx264",
                preset="medium",
                audio_codec="aac",
                fps=None,
                audio_bitrate="192k",
                logger=None,  # 禁用日志输出
            )
            
            # 清理临时视频
            if os.path.exists(input_video_path) and input_video_path != output_video_path:
                os.remove(input_video_path)
                
        except Exception as e:
            logger.error(f"Failed to add hold frames: {e}")
            # 如果后期处理失败，直接复制原始视频
            shutil.copy(input_video_path, output_video_path)
        

    async def merge_evolution_videos(self, video_paths: list[str]) -> str:
        """
        合并所有演变过渡视频成最终视频
        
        Returns:
            最终视频路径
        """
        
        if not video_paths:
            print("⚠️ No videos to merge")
            return ""
        
        final_video_path = os.path.join(self.working_dir, "evolution.mp4")
        
        if os.path.exists(final_video_path):
            print(f"🚀 Evolution video already exists")
            return final_video_path
        
        print(f"🎬 Merging {len(video_paths)} transition videos...")
        
        video_clips = []
        for video_path in video_paths:
            if os.path.exists(video_path):
                clip = VideoFileClip(video_path)
                video_clips.append(clip)
        
        if not video_clips:
            print("⚠️ No valid video clips to merge")
            return ""
        
        final_clip = concatenate_videoclips(video_clips, method="compose")
        final_clip.write_videofile(
            final_video_path, 
            codec="libx264", 
            preset="medium", 
            audio_codec="aac", 
            fps=None, 
            audio_bitrate="192k"
        )
        
        print(f"✅ Merged videos, saved to {final_video_path}")
        return final_video_path

    async def save_character_info(
        self, 
        glyph_png_paths: dict[str, str],
        hanzi: str | None = None,
    ):
        """
        保存汉字信息到 characters.json 和 character_portraits_registry.json
        
        Args:
            glyph_png_paths: 字形PNG路径字典
            hanzi: 要保存的汉字，默认使用 self.hanzi
        
        - characters.json: 保留原有角色，添加新汉字角色
        - character_portraits_registry.json: 按汉字组织，保留原有数据
        """
        target_hanzi = hanzi or self.hanzi
        
        # ===== 保存到 characters.json =====
        parent_dir = os.path.dirname(self.working_dir)
        characters_path = os.path.join(parent_dir, "characters.json")
        
        # 先读取已有的 characters.json，保留原有角色
        characters = []
        if os.path.exists(characters_path):
            with open(characters_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
        
        # 检查是否已存在该汉字角色
        existing_idx = None
        for char in characters:
            if char.get("identifier_in_scene") == f"{target_hanzi}字":
                existing_idx = char.get("idx")
                break
        
        # 如果不存在，添加新角色
        if existing_idx is None:
            new_idx = len(characters)
            characters.append({
                "idx": new_idx,
                "identifier_in_scene": f"{target_hanzi}字",
                "is_visible": True,
                "static_features": f"汉字-{target_hanzi}。",
                "dynamic_features": ""
            })
            char_idx = new_idx
        else:
            char_idx = existing_idx
        
        with open(characters_path, "w", encoding="utf-8") as f:
            json.dump(characters, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Saved characters info to {characters_path}")
        
        # ===== 保存到 character_portraits_registry.json =====
        registry_path = os.path.join(parent_dir, "character_portraits_registry.json")
        
        # 先读取已有的 registry，保留原有数据
        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        
        # 创建字形肖像目录，使用与 characters.json 一致的 idx
        char_portrait_dir = os.path.join(
            parent_dir, 
            "character_portraits", 
            f"{char_idx}_{target_hanzi}字"
        )
        os.makedirs(char_portrait_dir, exist_ok=True)
        
        # 添加该汉字的字形到 registry
        registry[f"{target_hanzi}字"] = {}
        
        for glyph_type, png_path in glyph_png_paths.items():
            if os.path.exists(png_path):
                dest_path = os.path.join(char_portrait_dir, f"{glyph_type}.png")
                shutil.copy(png_path, dest_path)
                registry[f"{target_hanzi}字"][glyph_type] = {
                    "path": dest_path,
                    "description": f"{glyph_type} of the character {target_hanzi}。{glyph_type_descriptions.get(glyph_type, '')}"
                }
        
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Saved character portraits registry to {registry_path}")

    async def crawl_relate_hanzi_info(self):
        """
        爬取相关文字的信息（只提取楷书图片），并保存到角色信息
        
        复用现有方法：
        - crawl_hanzi_info: 爬取信息
        - download_and_convert_svg_to_png: 下载并转换图片（指定只下载楷书）
        - save_character_info: 保存角色信息
        """
        if not self.relate_hanzi:
            return
        
        print(f"\n{'='*60}")
        print(f"🕷️ Crawling relate hanzi info: {self.relate_hanzi}")
        print(f"{'='*60}\n")
        
        for hanzi in self.relate_hanzi:
            print(f"\n--- Processing relate hanzi: {hanzi} ---")
            
            try:
                relate_dir = os.path.join(self.working_dir, "relate_hanzi", hanzi)
                os.makedirs(relate_dir, exist_ok=True)

                # 1. 爬取该汉字的信息
                hanzi_info = await self.crawl_hanzi_info(hanzi=hanzi, working_dir=relate_dir)
                
                if not hanzi_info.get("glyphs"):
                    print(f"⚠️ No glyphs found for {hanzi}")
                    continue
                
                # 2. 下载并转换字形图片（只下载楷书）
                glyph_png_paths = await self.download_and_convert_svg_to_png(
                    hanzi_info=hanzi_info,
                    working_dir=relate_dir,
                    target_types=["楷书"],  # 只下载楷书
                )
                
                if not glyph_png_paths:
                    print(f"⚠️ No PNG paths for {hanzi}")
                    continue
                
                # 3. 保存角色信息
                await self.save_character_info(
                    glyph_png_paths=glyph_png_paths,
                    hanzi=hanzi,
                )
                
                print(f"✅ Completed processing relate hanzi: {hanzi}")
                
            except Exception as e:
                logger.error(f"Failed to process relate hanzi {hanzi}: {e}")
                continue

    async def __call__(self) -> str:
        """
        执行完整的汉字演变流水线
        
        Returns:
            最终生成的视频路径
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting HanziPipeline for '{self.hanzi}'")
        print(f"{'='*60}\n")
        
        # Step 1: 爬取汉字信息
        hanzi_info = await self.crawl_hanzi_info()
        
        if self.check_interrupt("crawl"):
            return ""
        
        # Step 2: 下载并转换字形图片
        download_path = os.path.join(self.working_dir, "evolution")
        os.makedirs(download_path, exist_ok=True)
        glyph_png_paths = await self.download_and_convert_svg_to_png(hanzi_info, working_dir=download_path)
        
        if self.check_interrupt("download"):
            return ""
        
        # Step 3: 生成创意
        await self.generate_creative_idea(hanzi_info)
        
        if self.check_interrupt("idea"):
            return ""
        
        # Step 4: 生成过渡描述
        transitions = await self.generate_transitions(glyph_png_paths)
        
        if self.check_interrupt("transition"):
            return ""
        
        # return
        
        # Step 5: 生成演变动画
        video_paths = await self.generate_evolution_videos(glyph_png_paths, transitions)
        
        if self.check_interrupt("video"):
            return ""
        
        # Step 6: 合并视频
        final_video_path = await self.merge_evolution_videos(video_paths)

        # Step 7: 保存角色信息
        await self.save_character_info(glyph_png_paths)
        
        # Step 8: 爬取相关文字的信息（只提取楷书图片，并整理保存到角色信息）
        if self.relate_hanzi:
            await self.crawl_relate_hanzi_info()
        
        print(f"\n{'='*60}")
        print(f"✅ HanziPipeline completed!")
        print(f"   Final video: {final_video_path}")
        print(f"{'='*60}\n")
        
        return final_video_path

