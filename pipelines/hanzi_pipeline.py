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
import subprocess
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
from utils.audio import download_audio
from utils.provider_presets import create_chat_model
from utils.completion_logger import set_working_dir
from agents.hanzi_creative_agent import HanziCreativeAgent
from agents.hanzi_evolution_agent import HanziEvolutionAgent, EvolutionTransitions

logger = logging.getLogger(__name__)

glyph_type_descriptions = {
    "甲骨文": "如果未明确，泛指古人的汉字",
    "楷书": "如果未明确，泛指现在的汉字"
}
assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

def _ensure_url_protocol(url: str) -> str:
    """
    确保 URL 包含 http 协议。
    如果 URL 以 // 开头（无协议），自动添加 http://
    """
    if not url:
        return url
    if url.startswith("//"):
        return "https:" + url
    return url


class HanziPipeline:
    def __init__(
        self,
        chat_model: BaseChatModel,
        image_generator: Any,
        video_generator: Any,
        audio_generator: Any,
        minor_video_generator: Any,
        working_dir: str,
        hanzi: str,
        relate_hanzi: List[str] | None = None,
        interrupt_step: str | None = None,
        comfyui_enable: bool = True,
        disable_transition: bool = False,
    ):
        self.chat_model: BaseChatModel = chat_model
        self.image_generator: Any = image_generator
        self.video_generator: Any = video_generator
        self.audio_generator: Any = audio_generator
        self.minor_video_generator: Any = minor_video_generator
        self.working_dir: str = working_dir
        self.hanzi: str = hanzi
        self.relate_hanzi: List[str] | None = relate_hanzi
        self.interrupt_step: str | None = interrupt_step
        self.comfyui_enable: bool = comfyui_enable
        self.disable_transition = disable_transition
        
        os.makedirs(self.working_dir, exist_ok=True)
        self.temp_dir = os.path.join(self.working_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.working_dir, "character_portraits"), exist_ok=True)
        
        # 初始化 agents
        self.creative_agent: HanziCreativeAgent = HanziCreativeAgent(chat_model=chat_model)
        self.evolution_agent: HanziEvolutionAgent = HanziEvolutionAgent(chat_model=chat_model)

    @classmethod
    def init_from_config(cls, config_path: str):
        """从配置文件初始化"""
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        chat_model = create_chat_model(**config["chat_model"]["init_args"])
        backend = RenderBackend.from_config(config)
        hanzi = config.get("hanzi")

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            audio_generator=backend.audio_generator,
            working_dir=config.get("working_dir", f"./output/hanzi_{hanzi}"),
            hanzi=hanzi,
            interrupt_step=config.get("interrupt_step"),
            disable_transition=config.get("disable_transition", False),
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
        
        is_self = self.hanzi == hanzi
        save_path = os.path.join(working_dir, "hanzi_info.json")
        
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                hanzi_info = json.load(f)
            print(f"🚀 Loaded hanzi info from existing file.")
            return hanzi_info

        print(f"🕷️ Crawling hanzi info for '{hanzi}' from zdic.net...")
        
        url = f"https://www.zdic.net/hans/{hanzi}"
        
        try:
            soup = self._html_parse(url)
            
            # 提取基础解释
            basic_definitions = self._extract_basic_explanation(soup, is_self)
            
            # 提取字源字形
            glyphs = self._extract_glyphs(soup)
            
            hanzi_info = {
                "hanzi": hanzi,
                "basic_definitions": basic_definitions,
                "glyphs": glyphs,
                "source_url": url,
            }
            
            if is_self:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(hanzi_info, f, ensure_ascii=False, indent=4)
            
                print(f"✅ Crawled {hanzi} info and saved to {save_path}")

            return hanzi_info
            
        except Exception as e:
            logger.error(f"Failed to crawl hanzi info: {e}")
            raise

    def _html_parse(self, url: str) -> BeautifulSoup:
        """解析 HTML 页面"""

        if url:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            return BeautifulSoup(response.text, 'html.parser')

    def _extract_basic_explanation(self, soup: BeautifulSoup, is_self: bool = False) -> str:
        """提取基础解释"""
        
        if is_self:
            brush_stroke_path = os.path.join(self.working_dir, "brush_stroke.gif")
            
            if os.path.exists(brush_stroke_path):
                logger.info(f"Brush stroke image already exists: {brush_stroke_path}")
            else:
                # 下载笔画 gif。img id="bhbs"
                brush_stroke_img = soup.find('img', id='glyph-img')
                if brush_stroke_img:
                    brush_stroke_url = _ensure_url_protocol(brush_stroke_img["data-gif"])
                    download_image(brush_stroke_url, brush_stroke_path)

        definitions = []
        
        # 用户提供的 HTML 结构: div[data-type-block="基本解释"] > div.definitions > ol > li
        basic_block = soup.find('section', id='jbjs')
        if basic_block:
            definitions_div = basic_block.select('div.jbjs-reading')
            if definitions_div:
                for definition_div in definitions_div:
                    pinyin_head = definition_div.find('header', class_='jbjs-reading__head')
                    ol = definition_div.find('ol', class_='jbjs-list')
                    pinyin = ""

                    if pinyin_head:
                        pinyin = pinyin_head.find('span', class_='jbjs-reading__py').contents[0].strip()
                        audio_path = None
                    
                        if is_self:
                            # 下载拼音音频
                            temp_audio_path = os.path.join(self.working_dir, f"{pinyin}.mp3")
                            audio_path = temp_audio_path.replace(".mp3", ".flac")
                            if not os.path.exists(audio_path):
                                # 音频url在 a.audio_play_button data-src-mp3属性里
                                audio_url = _ensure_url_protocol(pinyin_head.find('button', class_='audio-btn')['data-audio'])
                            
                                if audio_url:
                                    download_audio(audio_url, temp_audio_path)
                                    subprocess.run(["ffmpeg", "-y", "-i", temp_audio_path,
                                                    "-ar", "44100",
                                                    "-ac", "2",
                                                    "-sample_fmt", "s16",
                                                    "-c:a", "flac",
                                                    audio_path], check=True)
                                    subprocess.run(["ffprobe", audio_path], check=True)
                                else:
                                    audio_path = None
                    
                    explanations = []
                    for li in ol.find_all('li'):
                        text = li.get_text(strip=True)
                        if text:
                            explanations.append(text)
                            
                    definitions.append({
                        "pinyin": pinyin,
                        "audio_path": audio_path or "",
                        "explanations": explanations,
                    })
        
        return definitions

    def _extract_glyphs(self, soup: BeautifulSoup, target_types: list[str] = ['甲骨文', '金文', '楚系简帛', '隶书', '楷书']) -> list[dict[str, str]]:
        """
        提取字源字形（甲骨文、金文、楚系簡帛、楷書）
        
        HTML结构:
        div[data-type-block="字源字形"] > div.definitions > table.zyyb
        
        Returns:
            字形列表，每个字形包含 type 和 svg_url
        """
        glyphs = []
        
        # 查找字源字形区域
        glyph_block = soup.find('section', id="zyzx")
        if not glyph_block:
            return glyphs
        
        table = glyph_block.find('div', class_='glyph-evolution')
        if not table:
            return glyphs
        
        # tbody = table.find('tbody')
        # if not tbody:
        #     return glyphs
        
        items = table.select('.glyph-evolution__item')
        if not items:
            return glyphs
        
        # # 第一行是表头，建立列索引到类型的映射
        # header_row = rows[0]
        # header_cells = header_row.find_all(['th', 'td'])
        # col_type_map = {}  # col_index -> type_name
        
        for idx, cell in enumerate(items):
            cell_text = cell.find('span', class_='glyph-evolution__label').get_text(strip=True)
            glyph_type = ""
            
            # 标准化类型名称
            if cell_text in ['甲骨文']:
                glyph_type = '甲骨文'
            elif cell_text in ['金文']:
                glyph_type = '金文'
            elif cell_text in ['楚系簡帛', '楚系簡帛', '楚簡', '楚簡文字']:
                glyph_type = '楚系简帛'
            elif cell_text in ['隶书']:
                glyph_type = '隶书'
            elif cell_text in ['楷書', '楷书']:
                glyph_type = '楷书'

            if glyph_type:
                if glyph_type not in target_types:
                    continue
                
                # 查找图片或链接
                img = cell.find('img')
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
                        # del col_type_map[col_idx]
        
        if not glyphs:
            # 请求楷书
            kaishu_img = soup.find('img', id='glyph-img')
            if kaishu_img:
                kaishu_img_url = _ensure_url_protocol(kaishu_img["src"])
                glyphs.append({
                    "type": "楷书",
                    "svg_url": kaishu_img_url,
                    "alt": "楷书",
                })

        if len(glyphs):
            # 提取繁体字链接
            # HTML结构:
            # div.char-card__variants
            #   span.meta-badge（内容包含"繁体"）
            #   紧跟其后的兄弟 ul.variant-list > li > a.variant-link[href]
            # 取 "繁体" badge 后第一个 variant-list 中唯一的 <a> 链接，拼接 host
            variants_div = soup.find('div', class_='char-card__variants')
            if variants_div:
                for child in variants_div.find_all(recursive=False):
                    if (
                        child.name == 'span'
                        and 'meta-badge' in (child.get('class') or [])
                        and '繁体' in child.get_text(strip=True)
                    ):
                        # 找其后紧跟的兄弟 ul.variant-list
                        variant_ul = child.find_next_sibling('ul', class_='variant-list')
                        if variant_ul:
                            link = variant_ul.find('a')
                            if link and link.get('href'):
                                href = link['href']
                                if href.startswith('//'):
                                    full_url = 'https:' + href
                                else:
                                    full_url = 'https://www.zdic.net' + href

                                # 获取繁体字形
                                fanti_soup = self._html_parse(full_url)
                                if fanti_soup:
                                    fanti_glyphs = self._extract_glyphs(fanti_soup, target_types=["楷书"])
                                    if fanti_glyphs:
                                        fanti_glyph = fanti_glyphs[0]
                                        fanti_glyph["type"] = "繁体"

                                        # 插入倒数第二位
                                        glyphs.insert(-1, fanti_glyph)

                        break

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
                    download_image(_ensure_url_protocol(svg_url), svg_path)
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
        
        # 临时文件路径（用于 cairosvg 转换）
        temp_png_path = png_path + ".temp.png"
        
        try:
            # 使用 cairosvg 转换
            cairosvg.svg2png(
                url=svg_path,
                write_to=temp_png_path,
                output_width=size,
                output_height=size,
                background_color=None
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
        # self._convert_transparent_black_to_white_bg(temp_png_path, png_path)
        # 将透明底黑字添加到米字格上
        mi_path = os.path.join(assets_dir, "mi.png")
        
        if os.path.exists(mi_path):
            # 打开米字格背景和透明底黑字
            mi_bg = Image.open(mi_path).convert("RGBA")
            char_img = Image.open(temp_png_path).convert("RGBA")
            
            # 将米字格缩放到目标尺寸 size x size
            mi_bg = mi_bg.resize((size, size), Image.Resampling.LANCZOS)
            
            # 缩放字符图片以适应米字格（保留一定边距，约占80%）
            char_resized = char_img.resize(
                (size, size), 
                Image.Resampling.LANCZOS
            )
            
            # 计算居中偏移
            x_offset = (size - char_resized.size[0]) // 2
            y_offset = (size - char_resized.size[1]) // 2
            
            # 将字符叠加到米字格上
            mi_bg.paste(char_resized, (x_offset, y_offset), mask=char_resized.split()[3])
            
            # 保存结果（转换为 RGB 去掉透明通道）
            mi_bg.convert("RGB").save(png_path, "PNG")
            print(f"✅ Added character to mi-grid background: {png_path}")
        else:
            # 米字格不存在，直接移动临时文件
            shutil.move(temp_png_path, png_path)
        
        # 清理临时文件
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)
    
    # def _convert_transparent_black_to_white_bg(self, input_path: str, output_path: str):
    #     """
    #     将透明底黑色字转换为白底黑色字
        
    #     Args:
    #         input_path: 输入图片路径（RGBA，可能有透明背景和黑色文字）
    #         output_path: 输出图片路径（RGB，白底黑字）
    #     """
    #     # 打开图片并转换为 RGBA
    #     img = Image.open(input_path).convert("RGBA")
        
    #     # 创建白色背景
    #     white_bg = Image.new("RGB", img.size, (255, 255, 255))
        
    #     # 将原图作为蒙版粘贴到白色背景上（保留黑色文字区域）
    #     white_bg.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为蒙版
        
    #     # 图片放大到 980 * 980
    #     white_bg = white_bg.resize((980, 980), Image.Resampling.LANCZOS)

    #     # 保存为 PNG
    #     white_bg.save(output_path, "PNG")
    #     print(f"✅ Converted to white background: {output_path}")

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
        glyph_order = ['甲骨文', '金文', '楚系简帛', '说文', '隶书', '繁体', '楷书']
        
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
            
            # # 合成带背景的图片路径
            # from_composite_path = os.path.join(evolution_dir, f"{from_type}_composite.png")
            # to_composite_path = os.path.join(evolution_dir, f"{to_type}_composite.png")
            # composite_prompt = "生成一个高质量的合成图片：将image2的黑色字符颜色调整为白色粉笔色，放在左侧米字格中（不要增减笔画），书写工整，大小合适，上下垂直居中。\nimage1右下角的小字去掉。"
            
            # # 如果合成图片已存在，跳过生成
            # if not os.path.exists(from_composite_path):
            #     bg_path = os.path.join(assets_dir, f"{from_type}.png")
            #     from_image = await self.image_generator.generate_single_image(
            #         prompt=composite_prompt,
            #         reference_image_paths=[bg_path, from_hanzi_path],
            #         size="2560x1440",
            #         workflow_name="qwen_edit",
            #     )
            #     from_image.save(from_composite_path)
            #     print(f"✅ Generated composite image for {from_type}")
            
            # if not os.path.exists(to_composite_path):
            #     if to_type == "楷书":
            #         composite_prompt = f"生成一个高质量的合成图片：将image2的黑色{self.hanzi}字颜色调整为白色粉笔色，放在左侧米字格中（不要增减笔画），书写工整，大小合适，上下垂直居中。\nimage1右下角的小字去掉。"
                
            #     bg_path = os.path.join(assets_dir, f"{to_type}.png")
            #     to_image = await self.image_generator.generate_single_image(
            #         prompt=composite_prompt,
            #         reference_image_paths=[bg_path, to_hanzi_path],
            #         size="2560x1440",
            #         workflow_name="qwen_edit",
            #     )
            #     to_image.save(to_composite_path)
            #     print(f"✅ Generated composite image for {to_type}")
            
            # 视频文件名：甲骨文_to_金文.mp4
            video_filename = f"{idx}_{from_type}_to_{to_type}.mp4"
            transition_video_path = os.path.join(evolution_dir, video_filename)
            transition_video_an_path = os.path.join(evolution_dir, f"{idx}_{from_type}_to_{to_type}_an.mp4")
            
            if os.path.exists(transition_video_an_path):
                print(f"🚀 Transition video already exists: {video_filename}")
                video_paths.append(transition_video_an_path)
                continue
            
            # 如果视频已存在，跳过
            if os.path.exists(transition_video_path):
                subprocess.run(["ffmpeg", "-i", transition_video_path, "-an", "-c:v", "copy", transition_video_an_path], check=True, capture_output=True)
                video_paths.append(transition_video_an_path)
                continue

            try:
                # 生成过渡视频（5秒纯动画，不包含停顿）
                if self.minor_video_generator:
                    prompt = f"开头1s保持首帧静止，1s后，{transition.description}，整个变化过程持续到4s，然后保持静止。没有背景音乐"
                    video_output = await self.minor_video_generator.generate_single_video(
                        prompt=prompt,
                        reference_image_paths=[from_hanzi_path, to_hanzi_path],
                        duration=5,
                        aspect_ratio="1:1"
                    )
                else:
                    prompt = f"这是一个汉字不同时期字形的演变动画，是物理轮廓的平滑形变，从图片1平滑形变过渡到图片2，禁止任何逐笔书写的顺序感，整个视频保持白色米格背景，禁止改变白色米格背景\n开头1s保持首帧静止，1s后，{transition.description}，整个变化过程持续到4s，然后保持静止。没有背景音乐"
                    video_output = await self.video_generator.generate_single_video(
                        prompt=prompt,
                        reference_image_paths=[from_hanzi_path, to_hanzi_path],
                        duration=5,
                        aspect_ratio="1:1"
                    )
                
                # 保存原始视频
                # raw_video_path = os.path.join(evolution_dir, f"{from_type}_to_{to_type}_raw.mp4")
                video_output.save(transition_video_path)
                subprocess.run(["ffmpeg", "-i", transition_video_path, "-an", "-c:v", "copy", transition_video_an_path], check=True, capture_output=True)
                
                video_paths.append(transition_video_an_path)
                print(f"✅ Generated transition video {from_type} to {to_type}, saved to {transition_video_an_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate transition video {from_type} to {to_type}: {e}")
        
        return video_paths

    async def generate_narration_audio(self, glyph_png_paths: dict[str, str]) -> str:
        """
        生成旁白音频
        
        使用字形名称生成旁白音频，按以下规则对齐：
        - 第一个字形旁白从 0.3s 开始
        - 最后一个字形旁白结束时刚好是总时长 - 0.3s
        - 其他字形旁白，按 idx * 5 - audio 时长 / 2 对齐
        
        Args:
            glyph_png_paths: 字形 PNG 路径字典
        
        Returns:
            生成的音频文件路径
        """

        output_path = os.path.join(self.temp_dir, "narration.flac")
        
        if os.path.exists(output_path):
            print(f"🚀 Narration audio already exists: {output_path}")
            return output_path
        
        print(f"🎬 Generating narration audio for {len(glyph_png_paths)} glyphs...")

        # 字形类型列表（按演变顺序）
        glyph_types = list(glyph_png_paths.keys())
        
        if len(glyph_types) < 2:
            print("⚠️ Not enough glyphs to generate narration audio")
            return ""
        
        # 总时长 = 5s * (字形数量 - 1)
        total_duration = 5 * (len(glyph_types) - 1)
        
        audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # 生成每个字形的旁白音频
        silence_1s = os.path.join(assets_dir, "1_s.flac")
        narration_info = []  # [(audio_path, duration, start_time)]
        current_duration = 0
        
        for idx, glyph_type in enumerate(glyph_types):
            # 生成旁白音频，prompt 就是字形名称
            audio_path = os.path.join(audio_dir, f"narration_{idx}.flac")
            
            if not os.path.exists(audio_path):
                print(f"🎙️ Generating narration for {glyph_type}...")
                audio_output = await self.audio_generator.generate_single_audio(
                    prompt=glyph_type,
                    character="旁白"
                )
                audio_output.save(audio_path)
            
            # 获取音频实际时长
            probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ]
            audio_duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())
            
            # 计算开始时间
            if idx == 0:
                # 第一个字形旁白从 0.3s 开始
                start_time = 0.3
            elif idx == len(glyph_types) - 1:
                # 最后一个字形旁白结束时要刚好是总时长 - 0.3s
                start_time = total_duration - current_duration - 0.3 - audio_duration
            else:
                # 其他字形旁白，按 idx * 5 - audio_duration / 2 对齐
                start_time = idx * 5 - 1 - current_duration
            
            current_duration = current_duration + audio_duration + start_time
            narration_info.append((audio_path, audio_duration, start_time))
            print(f"✅ Narration for {glyph_type}: duration={audio_duration:.2f}s, start={start_time:.2f}s")
            
        # 拼接音频，填充静音片段使总时长与视频一致
        # 使用 ffmpeg 合成音频（每个旁白前后填充静音）
        concat_parts = [
            "-i", silence_1s
        ]
        
        for i, (audio_path, audio_duration, start_time) in enumerate(narration_info):
            # 前置静音
            if start_time > 0:
                silence_before_path = os.path.join(audio_dir, f"silence_before_{i}.flac")
                subprocess.run([
                    "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", str(start_time), "-y", silence_before_path
                ], check=True, capture_output=True)
                concat_parts.extend(["-i", silence_before_path])
            
            concat_parts.extend(["-i", audio_path])
            
        concat_parts.extend(["-i", silence_1s])

        # 构建 ffmpeg 命令
        filter_complex = f"concat=n={int(len(concat_parts) / 2)}:v=0:a=1[out]"
        subprocess.run([
            "ffmpeg", "-y",
            *concat_parts,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "44100",
            output_path
        ], check=True, capture_output=True)
        
        return output_path
    
    def add_audio_to_video(
        self,
        video_path: str,
        audio_path: str,
        glyph_png_paths: dict[str, str],
        output_path: str
    ) -> str:
        """
        将旁白音频添加到视频中
        ffmpeg 合成视频，视频宽高 1344*736
            1. 取 assets_dir/blackboard.png 作为背景，铺满时长
            2. 取audio时长，视频前后静止1s，总时长为 1 + audio_duration + 1
            3. 音频从1s开始播放；视频从1s开始加入
            4. 将视频调整为 500 * 500；并放在 422, 186 位置
        
        Args:
            video_path: 视频路径
            audio_path: 音频路径
            output_path: 输出视频路径
        
        Returns:
            添加音频后的视频路径
        """

        if not video_path or not audio_path:
            return video_path

        if not os.path.exists(video_path):
            print(f"⚠️ Video file not found: {video_path}")
            return video_path

        if not os.path.exists(audio_path):
            print(f"⚠️ Audio file not found: {audio_path}")
            return video_path

        print("🎬 Adding narration audio to video...")

        # =========================
        # 获取音频时长
        # =========================
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        audio_duration = float(result.stdout.strip())
        print(f"Audio duration: {audio_duration:.2f}s")

        # =========================
        # 参数
        # =========================
        video_width = 1680
        video_height = 960

        blackboard_path = os.path.join(assets_dir, "blackboard.png")
        
        # 看我变
        segment_audio_path = os.path.join(assets_dir, "evolution.flac")
        segment_probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", segment_audio_path
        ]
        segment_audio_duration = float(subprocess.run(segment_probe_cmd, capture_output=True, text=True).stdout.strip())

        # 前1s + 音频 + 后1s
        total_duration = segment_audio_duration + audio_duration + 2
        
        final_audio_path = os.path.join(self.temp_dir, "evolution_final_audio.flac")
        
        if os.path.exists(final_audio_path):
            print(f"⚠️ Final audio already exists, skipping audio concatenation")
        else:        
            silence_1s = os.path.join(assets_dir, "1_s.flac")
            
            concat_parts = [
                "-i", segment_audio_path,
                "-i", silence_1s,
                "-i", audio_path,
                "-i", silence_1s,
            ]
            
            filter_complex = f"concat=n={int(len(concat_parts) / 2)}:v=0:a=1[out]"
            subprocess.run([
                "ffmpeg", "-y",
                *concat_parts,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", "44100",
                final_audio_path
            ], check=True, capture_output=True)

        # 视频区域
        video_size = 400
        video_x = 640
        video_y = 356

        # =========================
        # label 动画参数
        # =========================
        glyph_types = list(glyph_png_paths.keys())
        glyph_len = len(glyph_types)

        content_start = segment_audio_duration + 1.0
        content_end = total_duration - 1.0
        content_duration = content_end - content_start

        item_duration = content_duration / glyph_len

        x_center = "(w-text_w)/2"

        # macOS 中文字体
        font_path = "/System/Library/Fonts/PingFang.ttc"

        # =========================
        # 构建 filter_complex
        # =========================
        filter_parts = []

        # 背景
        filter_parts.append(
            f"[0:v]"
            f"scale={video_width}:{video_height}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={video_width}:{video_height}"
            f"[bg]"
        )

        # 视频
        filter_parts.append(
            f"[1:v]fps=25,scale={video_size}:{video_size},"
            f"setpts=PTS+{1+segment_audio_duration}/TB[video]"
        )

        # 1秒后显示视频
        filter_parts.append(
            f"[bg][video]"
            f"overlay={video_x}:{video_y}[v0]"
        )

        current_layer = "v0"

        # =========================
        # drawtext
        # =========================
        for i, glyph_type in enumerate(glyph_types):

            item_start = content_start + i * item_duration
            item_end = item_start + item_duration

            fade_duration = min(0.5, item_duration * 0.25)

            fade_in_end = item_start + fade_duration
            fade_out_start = item_end - fade_duration

            alpha_expr = (
                f"if(lt(t,{item_start}),0,"
                f"if(lt(t,{fade_in_end}),(t-{item_start})/{fade_duration},"
                f"if(lt(t,{fade_out_start}),1,"
                f"if(lt(t,{item_end}),({item_end}-t)/{fade_duration},0))))"
            )

            next_layer = f"v{i+1}"

            filter_parts.append(
                f"[{current_layer}]"
                f"drawtext="
                f"text='{glyph_type}':"
                f"fontfile='{font_path}':"
                f"fontsize=60:"
                f"fontcolor=white:"
                f"borderw=2:"
                f"bordercolor=black:"
                f"x={x_center}:"
                f"y=260:"
                f"alpha='{alpha_expr}'"
                f"[{next_layer}]"
            )

            current_layer = next_layer

        # 最终视频输出
        filter_parts.append(
            f"[{current_layer}]format=yuv420p[vout]"
        )

        # =========================
        # 音频
        # 前后各加1秒静音
        # =========================

        # # 音频重采样
        # filter_parts.append(
        #     "[2:a]aresample=44100[aout]"
        # )

        # # 音频尾部补1秒静音
        # filter_parts.append(
        #     "[a1]apad=pad_dur=1[aout]"
        # )

        filter_complex = ";".join(filter_parts)

        # =========================
        # ffmpeg
        # =========================
        cmd = [
            "ffmpeg",
            "-y",

            # 背景
            "-loop", "1",
            "-i", blackboard_path,

            # 视频
            "-i", video_path,

            # 音频
            "-i", final_audio_path,

            # filter
            "-filter_complex", filter_complex,

            # map
            "-map", "[vout]",
            "-map", "2:a",

            # 时长
            "-t", str(total_duration),

            # 视频编码
            "-c:v", "libx264",
            "-preset", "medium",

            # 音频编码
            "-c:a", "aac",
            "-b:a", "192k",

            # 输出格式
            "-pix_fmt", "yuv420p",

            # 结束
            "-shortest",

            output_path
        ]

        print(" ".join(cmd))

        try:
            subprocess.run(
                cmd,
                check=True
            )

            print(f"✅ Added audio to video: {output_path}")

        except subprocess.CalledProcessError as e:
            print("❌ ffmpeg error:")
            print(e)
            return video_path

        return output_path
        

    async def merge_evolution_videos(self, video_paths: list[str]) -> str:
        """
        合并所有演变过渡视频成最终视频
        
        Returns:
            最终视频路径
        """
        
        if not video_paths:
            print("⚠️ No videos to merge")
            return ""
        
        final_video_path = os.path.join(self.temp_dir, "evolution.mp4")
        
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
                    # target_types=["甲骨文", "楷书"],  # 只下载楷书
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

    async def generate_py_video(self, hanzi_info: dict[str, Any], glyph_png_paths: dict[str, str]) -> str:
        """
        生成拼音视频
        hanzi_info.basic_definitions 获取 pinyin 和 audio_path。生成数组（多音字）
        使用 image_generator 生成拼音图片，大小为 600*300
        取 assets_dir/blackboard.png 作为背景
        取 楷书的 png 图片为汉字图片
        按每个读音读两次，每次1s，中间间隔1s，多音字读音之间间隔2s算。拼接音频。
        总时长前后静止1s，所以总时长为 1 + (3 * len + 2 * (len - 1)) + 1
        ffmpeg 合成视频，视频宽高 1344*736
            1. 背景为 assets_dir/blackboard.png，铺满时长
            2. 汉字图片放在位置 472, 286 大小 400 * 400，前1s和后1s静止显示
            3. 拼音图片从1s开始显示。每个图片按大小 200 * 100
                如果是单音字，则居中，位置为 572, 150
                如果是多音字，则每个读音间隔 100；按顺序从左到右出现，出现后不消失。每个出现的时机为 1 + idx * 5
            4. 音频从1s开始播放
        """
        print(f"🎬 Starting generate pinyin video for '{self.hanzi}'...")

        output_video_path = os.path.join(self.temp_dir, "pinyin_video.mp4")

        if os.path.exists(output_video_path):
            print(f"⚠️ Output video already exists, skipping py video generation")
            return output_video_path

        basic_definitions = hanzi_info.get("basic_definitions", [])
        if not basic_definitions:
            print(f"⚠️ No basic definitions found, skipping py video generation")
            return ""
        
        # 创建临时目录
        temp_dir = os.path.join(self.temp_dir, "py_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 获取所有拼音
        pinyins = [d["pinyin"] for d in basic_definitions]
        audio_paths = [d["audio_path"] for d in basic_definitions]
        num_pinyins = len(pinyins)
        
        print(f"📝 Found {num_pinyins} pinyin(s): {pinyins}")
        
        # 资源路径
        blackboard_path = os.path.join(assets_dir, "blackboard.png")
        
        # 获取楷书图片路径
        kaishu_path = None
        for glyph_type, png_path in glyph_png_paths.items():
            if glyph_type == "楷书" and png_path:
                kaishu_path = png_path
                break
        
        if not kaishu_path or not os.path.exists(kaishu_path):
            print(f"⚠️ Kaishu image not found, skipping py video generation")
            return ""
        
        audio_durations = []
        # 计算内容时长：每个读音读两次（audio_duration+1s），中间间隔1s，多音字间隔2s
        content_duration = 2 * (num_pinyins - 1)
        
        # 跟我读
        segment_audio_path = os.path.join(assets_dir, "read.flac")
        segment_probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", segment_audio_path
        ]
        segment_audio_duration = float(subprocess.run(segment_probe_cmd, capture_output=True, text=True).stdout.strip())
        
        for audio_path in audio_paths:
            if audio_path and os.path.exists(audio_path):
                # 使用 ffprobe 获取音频时长
                probe_cmd = [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", audio_path
                ]
                audio_duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())
                audio_durations.append(audio_duration)
                content_duration += audio_duration + 1
        
        # 总时长 = 1s(前静止) + content_duration + 1s(后静止)
        total_duration = segment_audio_duration + 1 + content_duration + 1
        print(f"⏱️ Content duration: {content_duration}s, Total duration: {total_duration}s")
        
        # 拼接音频：每个读音读两次，每次1s，间隔1s，多音字间隔2s
        final_audio_path = os.path.join(temp_dir, "py_final_audio.flac")
        
        if os.path.exists(final_audio_path):
            print(f"⚠️ Final audio already exists, skipping audio concatenation")
        else:        
            silence_1s = os.path.join(assets_dir, "1_s.flac")
            silence_2s = os.path.join(assets_dir, "2_s.flac")
            # audio_list_file = os.path.join(temp_dir, "py_audio_list.txt")
            concat_parts = [
                "-i", segment_audio_path,
                "-i", silence_1s
            ]
            
            for idx, audio_path in enumerate(audio_paths):
                if audio_path and os.path.exists(audio_path):
                    concat_parts.extend([
                        "-i", audio_path,
                        "-i", silence_1s,
                        "-i", audio_path,
                    ])

                    # 多音字间隔2s（最后一个音后面不间隔）
                    if idx < num_pinyins - 1:
                        concat_parts.extend([
                            "-i", silence_2s,
                        ])
                        total_duration += 2
                        
            concat_parts.extend([
                "-i", silence_1s
            ])
                        
            # 使用 ffmpeg 拼接音频
            # concat_audio_path = os.path.join(temp_dir, "py_concat.flac")
            filter_complex = f"concat=n={int(len(concat_parts) / 2)}:v=0:a=1[out]"
            subprocess.run([
                "ffmpeg", "-y",
                *concat_parts,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", "44100",
                final_audio_path
            ], check=True, capture_output=True)
            
            # # 给音频前面加1s静音，使音频从1s开始播放
            # subprocess.run([
            #     "ffmpeg", "-y",
            #     "-i", silence_1s,
            #     "-i", concat_audio_path,
            #     "-i", silence_1s,
            #     "-filter_complex", "concat=n=3:v=0:a=1[out]",
            #     "-map", "[out]",
            #     "-ar", "44100",
            #     final_audio_path
            # ], check=True, capture_output=True)
        
        # 构建 ffmpeg 视频合成命令
        # 视频尺寸
        video_width = 1680
        video_height = 960
        
        # 拼音文字位置和样式（居中显示）
        pinyin_y = 260  # y 位置
        font_size = 80  # 字体大小
        
        # 计算每个拼音的显示时间
        # 每个读音读两次，中间间隔1s
        pinyin_times = []  # [(start, end), ...]
        current_time = segment_audio_duration + 1.0  # 从1s开始
        
        for idx, audio_duration in enumerate(audio_durations):
            # 读两次 + 中间间隔1s
            pinyin_duration = audio_duration * 2 + 1
            pinyin_times.append((current_time, current_time + pinyin_duration))
            current_time += pinyin_duration
            # 多音字之间间隔2s
            if idx < num_pinyins - 1:
                current_time += 2
        
        # 构建 ffmpeg filter_complex（使用 drawtext）
        filter_parts = []
        
        # 背景铺满
        filter_parts.append(f"[0:v]scale={video_width}:{video_height}:force_original_aspect_ratio=increase,crop={video_width}:{video_height}[bg]")
        
        # 汉字图片：位置 522, 286，大小 300x300
        filter_parts.append(f"[1:v]scale=400:400[hanzi]")
        filter_parts.append(f"[bg][hanzi]overlay=640:356[bg_with_hanzi]")
        
        # 使用 drawtext 显示拼音（白色，居中）
        # 居中位置 x=(w-text_w)/2
        x_center = "(w-text_w)/2"
        
        if num_pinyins == 1:
            # 单音字：从1s开始显示
            start_time, end_time = 0, total_duration
            filter_parts.append(
                f"[bg_with_hanzi]drawtext=text='{pinyins[0]}':fontsize={font_size}:fontcolor=white:"
                f"x={x_center}:y={pinyin_y}:enable='between(t\\,{start_time}\\,{end_time})'[bg_with_content]"
            )
            prev_bg = "[bg_with_content]"
        else:
            # 多音字：每个拼音独立显示，带2s淡入淡出过渡
            # 使用 fade 滤镜实现过渡效果
            all_py = "[bg_with_hanzi]"
            for idx, (pinyin, (start_time, end_time)) in enumerate(zip(pinyins, pinyin_times)):
                # fade_duration = 2.0
                # fade_start = end_time - fade_duration
                
                # 构建此拼音的显示逻辑：fade in 开始，solid 中间，fade out 结束
                filter_parts.append(
                    f"{all_py}drawtext=text='{pinyin}':fontsize={font_size}:fontcolor=white:"
                    f"x={x_center}:y={pinyin_y}:enable='between(t\\,{start_time}\\,{end_time})'"
                    f"[py_{idx}]"
                )
                
                # # 添加 fade 效果
                # if idx == 0:
                #     # 第一个：fade in
                #     filter_parts.append(f"[tmp_{idx}]fade=t=in:st={start_time}:d={fade_duration}[py_{idx}]")
                # elif idx == num_pinyins - 1:
                #     # 最后一个：fade out
                #     filter_parts.append(f"[tmp_{idx}]fade=t=out:st={fade_start}:d={fade_duration}[py_{idx}]")
                # else:
                #     # 中间的：fade in + fade out
                #     filter_parts.append(f"[tmp_{idx}]fade=t=in:st={start_time}:d={fade_duration/2}[py_{idx}_in]")
                #     filter_parts.append(f"[py_{idx}_in]fade=t=out:st={fade_start}:d={fade_duration}[py_{idx}]")
                
                all_py = f"[py_{idx}]"
            
            # 合并所有拼音层
            filter_parts.append(f"{all_py}format=yuv420p[bg_with_content]")
            prev_bg = "[bg_with_content]"
        
        filter_complex = ";".join(filter_parts)
        
        # 构建命令
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", blackboard_path,
            "-loop", "1", "-i", kaishu_path,
            "-filter_complex", filter_complex,
            "-i", final_audio_path,
            "-map", f"{prev_bg}",
            "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium", "-t", str(total_duration),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        
        print(f"🎬 Generating pinyin video with ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"⚠️ Failed to generate pinyin video: {result.stderr}")
            return ""
            
        print(f"✅ Generated pinyin video: {output_video_path}")
        return output_video_path
    
    async def generate_stroke_video(self) -> str:
        """
        生成笔画视频
        笔画 gif 为 working_dir/brush_stroke.gif
        获取 gif 时长
        总时长为前后静止1s，1 + gif_duration + 1
        使用 chat_model 生成笔画顺序文本。prompt 为 '生成汉字"日"的笔画顺序。只返回笔画顺序的文案，如竖、横、横'
        将笔画顺序文本生成音频，并调整跟 gif_duration 时长一致
        ffmpeg 合成视频，视频宽高 1344*736
            1. 取 assets_dir/blackboard.png 作为背景，铺满时长
            2. 取 assets_dir/mi.png 作为米字格子
            3. gif 从1s开始出现，持续 gif_duration 秒
            4. 将 mi.png 和 gif 调整为 500 * 500；并放在 422, 186 位置；gif 在 mi.png 的上面
            5. 音频从1s开始播放，时长调整为 gif_duration
        Returns:
            视频路径
        """
        print(f"🎬 Starting generate stroke video for '{self.hanzi}'...")
        
        # 输出路径
        output_video_path = os.path.join(self.temp_dir, "stroke_video.mp4")
        
        if os.path.exists(output_video_path):
            print(f"🚀 Skipped generating stroke video, already exists")
            return output_video_path
        
        # 笔画 gif 路径
        brush_stroke_gif_path = os.path.join(self.working_dir, "brush_stroke.gif")
        if not os.path.exists(brush_stroke_gif_path):
            print(f"⚠️ Brush stroke gif not found: {brush_stroke_gif_path}")
            return ""
        
        # 获取 gif 时长
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", brush_stroke_gif_path
        ]
        gif_duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        try:
            gif_duration = float(gif_duration_result.stdout.strip())
        except (ValueError, subprocess.CalledProcessError):
            print(f"⚠️ Failed to get gif duration, using default 3s")
            gif_duration = 3.0
            
        # 跟我写
        segment_audio_path = os.path.join(assets_dir, "write.flac")
        segment_probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", segment_audio_path
        ]
        segment_audio_duration = float(subprocess.run(segment_probe_cmd, capture_output=True, text=True).stdout.strip())
        
        # 总时长 = 1s(前静止) + gif_duration + 1s(后静止)
        total_duration = segment_audio_duration + 1 + gif_duration + 1
        print(f"⏱️ Gif duration: {gif_duration}s, Total duration: {total_duration}s")
        
        # 生成音频
        final_audio_path = os.path.join(self.temp_dir, "stroke_final_audio.flac")
        
        if os.path.exists(final_audio_path):
            print(f"🚀 Skipped generating stroke audio, already exists")
        else:
            if not self.comfyui_enable:
                return ""

            # 使用 chat_model 生成笔画顺序文本
            stroke_prompt = f'生成汉字"{self.hanzi}"的笔画顺序。只返回笔画顺序的文案，如竖、横、横'
            print(f"🧠 Generating stroke order text...")
            
            stroke_order_text = ""
            for retry in range(3):
                try:
                    response = await self.chat_model.ainvoke(stroke_prompt)
                    stroke_order_text = response.content.strip()
                    print(f"📝 Stroke order: {stroke_order_text}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to generate stroke order text (retry {retry+1}): {e}")
                    if retry == 2:
                        stroke_order_text = ""
            
            
            if not stroke_order_text:
                # 如果没有生成文字，使用默认文字
                stroke_order_text = f"汉字{self.hanzi}"
            
            stroke_audio_path = os.path.join(self.temp_dir, "stroke_audio.flac")
            audio_output = await self.audio_generator.generate_single_audio(
                prompt=stroke_order_text,
                character="旁白"
            )
            audio_output.save(stroke_audio_path)
            print(f"✅ Generated stroke audio: {stroke_audio_path}")
            
            # 获取音频时长
            audio_probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", stroke_audio_path
            ]
            audio_duration = float(subprocess.run(audio_probe_cmd, capture_output=True, text=True).stdout.strip())
            print(f"⏱️ Stroke audio duration: {audio_duration}s, Target duration: {gif_duration}s")
            
            # 将音频时长调整得跟 gif 时长一致
            # 使用 atempo 调整播放速度
            tempo_ratio = audio_duration / gif_duration
            adjusted_stroke_audio_path = os.path.join(self.temp_dir, "stroke_audio_adjusted.flac")
            subprocess.run([
                "ffmpeg", "-y", "-i", stroke_audio_path,
                "-filter:a", f"atempo={tempo_ratio}",
                adjusted_stroke_audio_path
            ], check=True, capture_output=True)
            stroke_audio_path = adjusted_stroke_audio_path
            
            silence_1s = os.path.join(assets_dir, "1_s.flac")
            
            concat_parts = [
                "-i", segment_audio_path,
                "-i", silence_1s,
                "-i", stroke_audio_path,
                "-i", silence_1s,
            ]
        
            # 使用 ffmpeg 拼接音频
            filter_complex = f"concat=n={int(len(concat_parts) / 2)}:v=0:a=1[out]"
            subprocess.run([
                "ffmpeg", "-y",
                *concat_parts,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", "44100",
                final_audio_path
            ], check=True, capture_output=True)
            
        # 资源路径
        blackboard_path = os.path.join(assets_dir, "blackboard.png")
        mi_grid_path = os.path.join(assets_dir, "mi.png")
        
        # 视频尺寸
        video_width = 1680
        video_height = 960
        
        # 米字格和 gif 尺寸及位置
        grid_size = 440
        grid_x = 620
        grid_y = 326
        
        # 构建 ffmpeg 命令
        # 1. 背景铺满时长
        # 2. mi.png 作为米字格子，500x500，放在 422, 186
        # 3. gif 从1s开始，放在 mi.png 上面（同一位置），持续 gif_duration
        # 4. 音频从1s开始播放（使用adelay延迟1s）
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", blackboard_path,
            "-loop", "1", "-i", mi_grid_path,
            "-i", brush_stroke_gif_path,
            "-filter_complex", (
                f"[0:v]scale={video_width}:{video_height}:force_original_aspect_ratio=increase,crop={video_width}:{video_height}[bg];"
                f"[1:v]scale={grid_size}:{grid_size}[mi_grid];"
                f"[2:v]fps=25,scale={grid_size}:{grid_size},"
                f"setpts=PTS+{1+segment_audio_duration}/TB[gif];"
                f"[bg][mi_grid]overlay={grid_x}:{grid_y}[bg_with_grid];"
                f"[bg_with_grid][gif]overlay={grid_x}:{grid_y}[video]"
            ),
            "-i", final_audio_path,
            # "-filter_complex", "[3:a]adelay=1000|1000[a_delayed]",
            "-map", "[video]",
            "-map", "3:a",
            "-t", str(total_duration),
            "-c:v", "libx264", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        
        print(f"🎬 Generating stroke video with ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ ffmpeg error: {result.stderr}")
            return ""
        
        print(f"✅ Generated stroke video: {output_video_path}")
        return output_video_path

    async def generate_evolution_video(self, glyph_png_paths: list[str]) -> str:
        """
        生成演变动画
        """
        
        print(f"🎬 Starting generate evolution video for '{self.hanzi}'...")
        
        final_video_path = os.path.join(self.temp_dir, "evolution_final_video.mp4")
        
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped generating evolution final video, already exists")
            return final_video_path
        
        # 生成过渡描述
        transitions = await self.generate_transitions(glyph_png_paths)
        
        if self.check_interrupt("transition"):
            return ""
        
        if self.comfyui_enable:
            video_paths = await self.generate_evolution_videos(glyph_png_paths, transitions)
            
            # 生成旁白
            narration_audio_path = await self.generate_narration_audio(glyph_png_paths)
            
            # Step 7: 合并视频
            merged_video_path = await self.merge_evolution_videos(video_paths)

            # Step 10: 将旁白音频添加到视频
            self.add_audio_to_video(merged_video_path, narration_audio_path, glyph_png_paths, final_video_path)
            
            return final_video_path

        return ""
    
    async def merge_videos(self, video_paths: list[str]) -> str:
        """
        合并视频。ffmpeg将视频合并成一个视频，并且使用 fade 转场
        
        Args:
            video_paths: 视频路径列表 [py_video, stroke_video, transition_video]
        
        Returns:
            合并后的视频路径
        """
        print(f"🎬 Merging hanzi videos...")
        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped generating final video, already exists")
            return final_video_path
        
        if not video_paths:
            print("⚠️ No videos to merge")
            return ""
        
        # 过滤存在的视频
        valid_video_paths = [v for v in video_paths if os.path.exists(v)]
        if not valid_video_paths:
            print("⚠️ No valid video files to merge")
            return ""
        
        print(f"⏱️ Merging {len(valid_video_paths)} videos with fade transitions...")
        
        # 创建临时文件列表
        concat_list_path = os.path.join(self.temp_dir, "final_concat_list.txt")
        
        try:
            # 获取每个视频的时长
            video_durations = []
            for video_path in valid_video_paths:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                     "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                    capture_output=True, text=True, check=True
                )
                duration = float(result.stdout.strip())
                video_durations.append(duration)
            
            # 生成 concat 列表文件
            with open(concat_list_path, "w") as f:
                for video_path in valid_video_paths:
                    f.write(f"file '{video_path}'\n")
            
            # 构建带有 fade 转场的 filter_complex
            # fade 转场：前一个视频淡出，后一个视频淡入
            # 使用 crossfade 效果
            filter_complex = ""
            
            if len(valid_video_paths) == 1:
                # 单个视频直接复制
                subprocess.run([
                    "ffmpeg", "-y", "-i", valid_video_paths[0],
                    "-c:v", "libx264", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "192k",
                    final_video_path
                ], check=True, capture_output=True)
            else:
                # 多个视频直接拼接（无转场）
                inputs = []
                for i, video_path in enumerate(valid_video_paths):
                    inputs.extend(["-i", video_path])
                
                # 使用 concat 滤镜直接拼接
                concat_inputs = ""
                for i in range(len(valid_video_paths)):
                    concat_inputs += f"[{i}:v][{i}:a]"
                
                filter_complex = f"{concat_inputs}concat=n={len(valid_video_paths)}:v=1:a=1[outv][outa]"
                
                cmd = ["ffmpeg", "-y"] + inputs + [
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]",
                    "-c:v", "libx264", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "192k",
                    final_video_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
            
            # 清理临时文件
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            
            print(f"✅ Merged videos with fade transitions: {final_video_path}")
            return final_video_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to merge videos: {e}")
            # 清理临时文件
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            return ""


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
        
        final_video_path = ""

        # # Step4: 生成拼音视频
        py_video_path = await self.generate_py_video(hanzi_info, glyph_png_paths)

        # Step5: 生成笔画视频
        stroke_video_path = await self.generate_stroke_video()

        # Step 6: 生成演变动画
        transition_video_path = ""

        if not self.disable_transition and len(glyph_png_paths) > 2:
            transition_video_path = await self.generate_evolution_video(glyph_png_paths)

        if self.check_interrupt("transition"):
            return ""
        
        # Step 7: 合并视频
        final_video_path = await self.merge_videos([py_video_path, stroke_video_path, transition_video_path])

        if self.check_interrupt("video"):
            return ""

        # Step 8: 保存角色信息
        await self.save_character_info(glyph_png_paths)
        
        # Step 9: 爬取相关文字的信息（只提取楷书图片，并整理保存到角色信息）
        if self.relate_hanzi:
            await self.crawl_relate_hanzi_info()
        
        print(f"\n{'='*60}")
        print(f"✅ HanziPipeline completed!")
        print(f"   Final video: {final_video_path}")
        print(f"{'='*60}\n")
        
        return final_video_path

