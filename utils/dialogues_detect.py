"""
通过 fast-vad + 语音周期性二次校验，检测视频中的对话时间区间。

fast-vad 基于 log-energy band 特征分类，对背景音乐、环境音已有一定抑制，
但视频生成场景中的机械变形音效（电子合成音、脉冲音等）仍会被误判为语音。

本模块在 fast-vad 粗筛后增加二次校验：
  1. 逐帧计算归一化自相关（周期强度）与基频 f0；
  2. 周期强度高且 f0 落在人类发声范围 [70, 400]Hz 的帧视为"语音帧"；
  3. 按 fast-vad 候选段统计语音帧密度：机械音效稀疏（<35%），真声密集；
  4. 密度不足但簇内存在时长 >= 0.5s 的"连续语音串"（相邻语音帧无断帧）
     的段予以保底保留，避免"全程背景音 + 短人声"场景误杀。
"""
import os
import subprocess
import tempfile
import logging
from typing import List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000


def get_video_duration(video_path: str) -> float:
    """通过 ffprobe 获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_audio_to_wav(video_path: str) -> str | None:
    """用 ffmpeg 提取 16kHz 单声道 wav，返回临时文件路径（调用方负责清理）"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", str(VAD_SAMPLE_RATE),
             "-f", "wav", tmp.name],
            capture_output=True, text=True, check=True,
        )
        return tmp.name
    except Exception as e:
        logger.warning(f"ffmpeg extract audio failed for {video_path}: {e}")
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return None


def detect_dialogues_from_video(
    video_path: str,
    *,
    mode: str = "aggressive",
    threshold_probability: float | None = None,
    min_speech_ms: int = 100,
    min_silence_ms: int = 300,
    hangover_ms: int = 100,
    periodicity_threshold: float = 0.45,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    min_voice_ratio: float = 0.35,
    min_cluster_s: float = 0.3,
    long_cluster_s: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    检测视频中的对话区间（fast-vad 粗筛 + 语音周期性二次校验）。

    流程：
      1. fast-vad 粗筛出候选语音段；
      2. 对每段逐帧计算周期强度与基频，周期强度 >= periodicity_threshold
         且 f0 在 [f0_min, f0_max] 内的帧判定为语音帧（人类浊音帧）；
      3. 段内语音帧重聚合为语音簇（簇间 gap <= 0.25s 合并，簇长 < min_cluster_s 丢弃）；
      4. 保留条件：段内语音帧密度 >= min_voice_ratio（机械音效稀疏被剔除），
         或存在时长 >= long_cluster_s 的连续语音簇（防止误杀"背景音+短人声"）。

    Args:
        video_path: 视频文件路径
        mode: fast-vad 内置模式，permissive / normal / aggressive。
              默认 aggressive（低误报率，减少背景音误判）
        threshold_probability / min_speech_ms / min_silence_ms / hangover_ms:
              fast-vad 自定义调参，传 None 时使用内置模式默认值
        periodicity_threshold: 归一化自相关峰值阈值，越高越严格（>0.45 更偏向强浊音）
        f0_min / f0_max: 人类发声基频范围，用于排除非人声周期信号
        min_voice_ratio: 候选段内语音帧最小占比，低于该值视为背景音/音效误报
        min_cluster_s: 语音簇最小时长，短于该值的零星音效被丢弃
        long_cluster_s: 保底保留的长语音簇阈值，用于"全程背景音+短人声"场景

    Returns:
        [(start, end), ...] 列表，单位为秒
    """
    if not os.path.exists(video_path):
        return []

    wav_path = _extract_audio_to_wav(video_path)
    if not wav_path:
        return []

    try:
        import fast_vad

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转为单声道
        if sr != VAD_SAMPLE_RATE:
            # 理论上 ffmpeg 已重采样为 16kHz，此处兜底
            audio = _resample_audio(audio, sr, VAD_SAMPLE_RATE)
            sr = VAD_SAMPLE_RATE

        if threshold_probability is not None:
            vad = fast_vad.VAD.with_config(
                sr,
                threshold_probability=threshold_probability,
                min_speech_ms=min_speech_ms,
                min_silence_ms=min_silence_ms,
                hangover_ms=hangover_ms,
            )
        else:
            mode_map = {
                "permissive": fast_vad.mode.permissive,
                "normal": fast_vad.mode.normal,
                "aggressive": fast_vad.mode.aggressive,
            }
            vad = fast_vad.VAD.with_mode(sr, mode_map.get(mode, fast_vad.mode.aggressive))

        # detect_segments 返回 (N, 2) uint64 数组，每行为 [start, end] 样本索引（半开区间）
        segments = vad.detect_segments(audio)
        if len(segments) == 0:
            return []

        dialogues = _refine_with_periodicity(
            audio,
            sr,
            segments,
            periodicity_threshold=periodicity_threshold,
            f0_min=f0_min,
            f0_max=f0_max,
            min_voice_ratio=min_voice_ratio,
            min_cluster_s=min_cluster_s,
            long_cluster_s=long_cluster_s,
        )
        return dialogues
    except Exception as e:
        logger.warning(f"fast-vad detection failed for {video_path}: {e}")
        return []
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def _refine_with_periodicity(
    audio: np.ndarray,
    sr: int,
    segments,
    *,
    periodicity_threshold: float,
    f0_min: float,
    f0_max: float,
    min_voice_ratio: float,
    min_cluster_s: float,
    long_cluster_s: float,
) -> List[Tuple[float, float]]:
    """
    对 fast-vad 候选段做语音周期性二次校验，返回过滤后的对话区间。
    """
    frame_win = int(0.1 * sr)
    frame_hop = int(0.05 * sr)
    total = len(audio)
    # 每帧标记 [start_sample, end_sample, is_voice]
    frames = []
    t = 0
    while t + frame_win <= total:
        x = audio[t:t + frame_win]
        is_voice, _ = _is_speech_frame(
            x, sr,
            periodicity_threshold=periodicity_threshold,
            f0_min=f0_min, f0_max=f0_max,
        )
        frames.append((t, t + frame_win, is_voice))
        t += frame_hop

    out: List[Tuple[float, float]] = []
    for seg_start, seg_end in segments:
        seg_lo, seg_hi = int(seg_start), int(seg_end)
        # 落在段内的帧
        in_seg = [(s, e, v) for s, e, v in frames if s < seg_hi and e > seg_lo]
        if not in_seg:
            continue
        n_frames = len(in_seg)
        n_voice = sum(1 for _, _, v in in_seg if v)
        ratio = n_voice / n_frames

        # 段内语音帧重聚合为语音簇
        clusters = _merge_voice_frames(in_seg, min_cluster_s=min_cluster_s)

        # 保留条件 1：语音帧密度足够（真声密集）
        # 保留条件 2：簇内存在足够长的"连续语音串"（背景音+短人声保底）。
        #   连续语音串要求相邻语音帧之间无断帧，机械音效帧稀疏断开，
        #   即使 gap 合并成簇也无法凑出长连续串。
        keep_clusters = []
        for cs, ce, max_run_s in clusters:
            if ratio >= min_voice_ratio or max_run_s >= long_cluster_s:
                keep_clusters.append((cs, ce))
        out.extend(keep_clusters)

    out.sort(key=lambda x: x[0])
    return out


def _is_speech_frame(
    x: np.ndarray,
    sr: int,
    *,
    periodicity_threshold: float,
    f0_min: float,
    f0_max: float,
) -> Tuple[bool, float]:
    """单帧判定：RMS 足够 + 周期强度 >= 阈值 + f0 在人类发声范围"""
    rms = np.sqrt(np.mean(x**2))
    if rms < 0.005:
        return False, 0.0
    ps, f0 = _pitch_strength(x, sr)
    return (ps >= periodicity_threshold and f0_min <= f0 <= f0_max), ps


def _pitch_strength(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> Tuple[float, float]:
    """
    归一化自相关求周期强度与基频。
    返回 (周期强度, f0)，周期强度为自相关最大峰值（1 为完全周期）。
    """
    x = x - x.mean()
    n = len(x)
    lag_max = int(sr / fmin)
    lag_min = int(sr / fmax)
    if lag_max >= n:
        return 0.0, 0.0
    ac = np.correlate(x, x, "full")[n - 1:n + lag_max]
    ac /= (ac[0] + 1e-12)
    seg = ac[lag_min:lag_max]
    if len(seg) == 0:
        return 0.0, 0.0
    lag = np.argmax(seg) + lag_min
    return float(ac[lag]), float(sr / lag)


def _merge_voice_frames(
    in_seg,
    *,
    min_cluster_s: float,
    max_gap_s: float = 0.25,
) -> List[Tuple[float, float, float]]:
    """
    把语音帧合并为语音簇。

    簇间 gap 超过 max_gap_s 则断开，簇长 < min_cluster_s 丢弃。
    返回 [(start_s, end_s, max_run_s), ...]：
      max_run_s 为簇内"最长连续语音串"时长——相邻语音帧（start 间隔约一帧 hop）
      连续无断帧的时长，用于区分连续浊音（真声）与稀疏音效帧。
    """
    hop_s = 0.05  # 帧 hop，与 _refine_with_periodicity 中一致
    clusters = []
    cur_s = cur_e = None
    cur_run = 0.0  # 当前连续语音串时长
    max_run = 0.0  # 簇内最长连续语音串时长
    last_voice_start = None
    for s, e, is_voice in in_seg:
        if is_voice:
            if cur_s is None:
                cur_s, cur_e = s, e
                cur_run, max_run = hop_s, hop_s
            else:
                if s - last_voice_start <= hop_s * VAD_SAMPLE_RATE * 1.5:
                    cur_run += hop_s
                else:
                    cur_run = hop_s
                if cur_run > max_run:
                    max_run = cur_run
                if s - cur_e <= max_gap_s * VAD_SAMPLE_RATE:
                    cur_e = e
                else:
                    if cur_e - cur_s >= min_cluster_s * VAD_SAMPLE_RATE:
                        clusters.append((cur_s / VAD_SAMPLE_RATE, cur_e / VAD_SAMPLE_RATE, max_run))
                    cur_s, cur_e = s, e
                    cur_run, max_run = hop_s, hop_s
            last_voice_start = s
        else:
            if cur_s is not None and s - cur_e > max_gap_s * VAD_SAMPLE_RATE:
                if cur_e - cur_s >= min_cluster_s * VAD_SAMPLE_RATE:
                    clusters.append((cur_s / VAD_SAMPLE_RATE, cur_e / VAD_SAMPLE_RATE, max_run))
                cur_s, cur_e = None, None
                cur_run, max_run = 0.0, 0.0
    if cur_s is not None and cur_e - cur_s >= min_cluster_s * VAD_SAMPLE_RATE:
        clusters.append((cur_s / VAD_SAMPLE_RATE, cur_e / VAD_SAMPLE_RATE, max_run))
    return clusters


def _resample_audio(audio, src_sr: int, dst_sr: int):
    """简单的线性插值重采样（仅兜底用，正常流程 ffmpeg 已重采样）"""
    if src_sr == dst_sr:
        return audio
    n_out = int(len(audio) * dst_sr / src_sr)
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio)
