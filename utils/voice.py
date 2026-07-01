"""
音色管理器 (VoiceManager)

管理角色音色的注册与查询。支持：
- register(name, gender, voice_file)  注册角色音色
- get(name, gender)                  按角色名/性别获取音色
- 单例模式，全局共享同一份注册表
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ==================================================
# 数据模型
# ==================================================

@dataclass
class VoiceEntry:
    """单个音色条目"""
    voice_file: str                                    # 音色文件名（不含扩展名），如 "字博士"
    gender: str = "Male"                               # 角色性别: "Male" / "Female"
    display_name: str = ""                             # 可选展示名

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.voice_file


# ==================================================
# VoiceManager
# ==================================================

class VoiceManager:
    """
    音色管理器（单例）

    用法::

        vm = VoiceManager.instance()

        # 注册
        vm.register("字博士",  gender="Male",   voice_file="字博士")
        vm.register("小豆丁",  gender="Female", voice_file="小豆丁")
        vm.register("旁白",    gender="Male",   voice_file="旁白")

        # 设置性别默认音色
        vm.set_default_voice("Male",   "男生")
        vm.set_default_voice("Female", "女生")
        vm.set_fallback_voice("男生")          # 最终兜底

        # 查询
        vm.get("字博士")                       # → "字博士"
        vm.get(name=None,  gender="Female")    # → "女生"
        vm.get(name="未知", gender=None)       # → "男生"（兜底）
    """

    _instance: Optional[VoiceManager] = None

    def __init__(self):
        self._voices: Dict[str, VoiceEntry] = {}       # key = 角色名
        self._gender_defaults: Dict[str, str] = {}     # key = gender ("Male"/"Female"), value = voice_file
        self._fallback_voice: str = "男生"             # 最终兜底音色

    # ---------- 单例 ----------

    @classmethod
    def instance(cls) -> VoiceManager:
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（测试用）"""
        cls._instance = None

    # ---------- 注册 ----------

    def register(
        self,
        name: str,
        *,
        gender: str = "Male",
        voice_file: Optional[str] = None,
        display_name: str = "",
    ) -> None:
        """
        注册一个角色音色

        Args:
            name:        角色名（查询时的 key）
            gender:      性别，"Male" 或 "Female"
            voice_file:  音色文件名（不含扩展名），默认与 name 相同
            display_name: 展示名
        """
        vf = voice_file or name
        entry = VoiceEntry(voice_file=vf, gender=gender, display_name=display_name or name)
        self._voices[name] = entry
        logger.info(f"Registered voice: '{name}' → '{vf}' (gender={gender})")

    def set_default_voice(self, gender: str, voice_file: str) -> None:
        """设置某个性别的默认音色"""
        self._gender_defaults[gender] = voice_file
        logger.info(f"Set default voice for gender '{gender}': '{voice_file}'")

    def set_fallback_voice(self, voice_file: str) -> None:
        """设置最终兜底音色（当按名字和性别都匹配不到时使用）"""
        self._fallback_voice = voice_file
        logger.info(f"Set fallback voice: '{voice_file}'")

    # ---------- 查询 ----------

    def get(self, name: Optional[str] = None, *, gender: Optional[str] = None) -> str:
        """
        获取音色文件名

        优先级：
        1. name 精确匹配已注册的角色 → 返回该角色音色
        2. gender 匹配性别默认音色       → 返回性别默认音色
        3. 以上都不满足                   → 返回 fallback_voice

        Args:
            name:   角色名
            gender: 性别 ("Male" / "Female")

        Returns:
            音色文件名（不含扩展名）
        """
        # 1. 按角色名精确匹配
        if name and name in self._voices:
            entry = self._voices[name]
            logger.debug(f"Voice lookup by name '{name}': '{entry.voice_file}'")
            return entry.voice_file

        # 2. 按性别默认值
        if gender and gender in self._gender_defaults:
            vf = self._gender_defaults[gender]
            logger.debug(f"Voice lookup by gender '{gender}': '{vf}'")
            return vf

        # 3. 兜底
        logger.debug(f"Voice lookup fallback: '{self._fallback_voice}'")
        return self._fallback_voice

    def get_detailed(self, name: Optional[str] = None, *, gender: Optional[str] = None) -> VoiceEntry:
        """
        同 get() 但返回完整 VoiceEntry。
        兜底时构造一个临时 VoiceEntry。
        """
        if name and name in self._voices:
            return self._voices[name]

        vf = self.get(name=name, gender=gender)
        g = gender or self._voices[name].gender if name and name in self._voices else "Male"
        return VoiceEntry(voice_file=vf, gender=g)

    # ---------- 批量操作 ----------

    def register_batch(self, entries: Dict[str, Dict]) -> None:
        """
        批量注册

        entries 格式::

            {
                "字博士": {"gender": "Male",   "voice_file": "字博士"},
                "小豆丁": {"gender": "Female", "voice_file": "小豆丁"},
            }
        """
        for name, info in entries.items():
            self.register(
                name,
                gender=info.get("gender", "Male"),
                voice_file=info.get("voice_file"),
                display_name=info.get("display_name", name),
            )

    def list_all(self) -> Dict[str, VoiceEntry]:
        """列出所有已注册的角色音色"""
        return dict(self._voices)


# ==================================================
# 模块级便捷函数
# ==================================================

def register_voice(name: str, *, gender: str = "Male", voice_file: Optional[str] = None) -> None:
    """快捷注册"""
    VoiceManager.instance().register(name, gender=gender, voice_file=voice_file)


def get_voice(name: Optional[str] = None, *, gender: Optional[str] = None) -> str:
    """快捷查询"""
    return VoiceManager.instance().get(name, gender=gender)


# ==================================================
# 默认初始化：从旧 register_voices 迁移
# ==================================================

def _init_from_legacy():
    """从旧 config 加载初始音色"""
    try:
        from configs.config import register_voices as legacy
    except ImportError:
        return

    vm = VoiceManager.instance()
    for key, vf in legacy.items():
        if key == "default":
            vm.set_fallback_voice(vf)
        elif key in ("Male", "Female"):
            vm.set_default_voice(key, vf)
        else:
            # 推断性别：Female 默认值里有"女"相关字符则女，否则男
            gender = "Female" if ("女" in vf or "female" in vf.lower()) else "Male"
            vm.register(key, gender=gender, voice_file=vf)


_init_from_legacy()
