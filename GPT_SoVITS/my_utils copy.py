import io
import tempfile
import ffmpeg
import numpy as np
import torchaudio


# def load_audio(file, sr):
#     try:
#         # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
#         # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
#         # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
#         with tempfile.NamedTemporaryFile(delete=True) as temp_file:
#             temp_file.write(file)
#             temp_file.flush()  # 确保数据写入
#             out, _ = (
#                 ffmpeg.input(temp_file.name, threads=0)
#                 .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
#                 .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
#             )
#     except Exception as e:
#         raise RuntimeError(f"Failed to load audio: {e}")

#     return np.frombuffer(out, np.float32).flatten()

def load_audio(file, sr = None):
    """
    Load an audio from bytes and read as mono waveform, resampling as necessary
    
    Args:
        file: the input audio file
        sr: the sample rate of the audio file
    
    Returns:
        audio: the audio waveform
        sample_rate: the sample rate of the audio file, if sr is not None, the sample rate will be resampled to sr
    """
    audio, sample_rate = torchaudio.load(file)
    # 单声道
    audio = audio.mean(dim=0)
    # 降采样
    if sr != sample_rate and sr is not None:
        sample_rate = sr
        audio = torchaudio.functional.resample(audio, sample_rate, sr)
    return audio, sample_rate

import io
import numpy as np
from subprocess import Popen, PIPE
from concurrent.futures import ThreadPoolExecutor
import threading

import torchaudio

# 全局线程池，线程数设置为 8（可以根据实际性能测试调整）
# THREAD_POOL = ThreadPoolExecutor(max_workers=12)

# def load_audio(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
#     """
#     Load an audio from bytes and read as mono waveform, resampling as necessary
#     """
#     cmd = [
#         "ffmpeg",
#         "-nostdin",
#         "-threads", "0",
#         "-i", "pipe:0",
#         "-f", "s16le",
#         "-ac", "1",
#         "-acodec", "pcm_s16le",
#         "-ar", str(sr),
#         "pipe:1"
#     ]
    
#     try:
#         process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
#         out, err = process.communicate(input=audio_bytes)
#         if process.returncode != 0:
#             raise RuntimeError(f"Failed to load audio: {err.decode()}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to process audio: {str(e)}") from e

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# def load_audio_batch(audio_bytes_list: list, sr: int = 16000) -> list:
#     """
#     Load a batch of audio bytes in parallel
#     """
#     return list(THREAD_POOL.map(load_audio, audio_bytes_list, [sr]*len(audio_bytes_list)))

# # 线程本地存储，用于缓存每个线程的 FFmpeg 进程
# thread_local = threading.local()

# def get_ffmpeg_process(sr: int):
#     if not hasattr(thread_local, 'ffmpeg_process') or thread_local.ffmpeg_process.poll() is not None:
#         cmd = [
#             "ffmpeg",
#             "-nostdin",
#             "-threads", "0",
#             "-i", "pipe:0",
#             "-f", "s16le",
#             "-ac", "1",
#             "-acodec", "pcm_s16le",
#             "-ar", str(sr),
#             "pipe:1"
#         ]
#         thread_local.ffmpeg_process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     return thread_local.ffmpeg_process

# def load_audio_optimized(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
#     """
#     Optimized version of load_audio using thread-local FFmpeg processes
#     """
#     process = get_ffmpeg_process(sr)
#     try:
#         out, err = process.communicate(input=audio_bytes)
#         if process.returncode != 0:
#             raise RuntimeError(f"Failed to load audio: {err.decode()}")
#     except Exception as e:
#         # 如果出现错误，确保下次会创建新的进程
#         if hasattr(thread_local, 'ffmpeg_process'):
#             del thread_local.ffmpeg_process
#         raise RuntimeError(f"Failed to process audio: {str(e)}") from e

#     # 每次处理完后，创建新的进程以备下次使用
#     thread_local.ffmpeg_process = get_ffmpeg_process(sr)

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# def load_audio_batch_optimized(audio_bytes_list: list, sr: int = 16000) -> list:
#     """
#     Load a batch of audio bytes in parallel using the optimized version
#     """
#     return list(THREAD_POOL.map(load_audio_optimized, audio_bytes_list, [sr]*len(audio_bytes_list)))
    


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
