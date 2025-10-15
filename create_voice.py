import io
import os
import pickle
from bson import Binary
import zlib
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
import numpy as np
from openai import OpenAI
from opencc import OpenCC
from pymongo import MongoClient
import torch
from pyloudnorm import Meter
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import soundfile as sf

# 加载.env文件
load_dotenv(dotenv_path="./.env")
#load_dotenv(dotenv_path="/workspaces/.env")

app = FastAPI()

# MongoDB 连接配置（与 gateway.py 保持一致）
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# 从环境变量中获取其他配置
openai_api_key = os.getenv("OPENAI_API_KEY")

# 构建 MongoDB 连接字符串
CONNECTION_STRING = f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/voice?authSource=admin"

# 加载TTS配置


# 初始化OpenAI客户端
client = OpenAI(api_key=openai_api_key)

# 初始化MongoDB客户端
mongo_client = MongoClient(CONNECTION_STRING)
db = mongo_client.voice  # 数据库名
collection = db.voice_cache  # 集合名（与 gateway.py 保持一致）

config_path = "./GPT_SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
# 初始化TTS模型
tts = TTS(tts_config)

# 初始化OpenCC转换器
cc = OpenCC("t2s")  # 繁体到简体


def simplify_chinese(text: str) -> str:
    """
    将繁体中文转换为简体中文
    """
    return cc.convert(text)

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def balance_loudness(audio, target_loudness=-23.0):
    meter = Meter(32000)  # 创建一个响度计量器

    # 确保audio是NumPy数组
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().numpy()

    # 测量积分响度
    loudness = meter.integrated_loudness(audio)

    # 计算增益
    gain_db = target_loudness - loudness
    gain_linear = 10 ** (gain_db / 20.0)

    # 应用增益
    balanced_audio = audio * gain_linear

    # 应用软限幅以防止削波
    balanced_audio = np.tanh(balanced_audio)

    return balanced_audio

@app.post("/create_voice")
async def create_voice(id: str, file: UploadFile = File(...)):
    # 读取上传的音频文件
    audio_data = await file.read()

    # 将字节数据转换为numpy数组
    audio_np, sample_rate = sf.read(io.BytesIO(audio_data))

    # 应用响度平衡
    balanced_audio = balance_loudness(audio_np)

    # 将平衡后的音频转回字节数据
    balanced_audio_bytes = io.BytesIO()
    sf.write(balanced_audio_bytes, balanced_audio, sample_rate, format='WAV')
    balanced_audio_data = balanced_audio_bytes.getvalue()

    # 使用OpenAI API进行语音转文字
    audio_file = io.BytesIO(balanced_audio_data)
    audio_file.name = "audio.wav"  # OpenAI需要文件名
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    # 简体化文本
    prompt_text = simplify_chinese(transcript.text)

    # 处理音频和文本，获取必要的特征
    ref_audios_batch = [[balanced_audio_data]]  # 使用平衡后的音频数据
    prompt_texts_batch = [[prompt_text]]
    prompt_lang = "auto"  # 假设输入总是中文，你可以根据需要修改

    prompt_semantic, refer_spec = tts.set_ref_audio(ref_audios_batch, [0])
    refer_spec = refer_spec[0][0]
    refer_lengths = torch.LongTensor([refer_spec.size(2)]).to(refer_spec.device)
    refer_mask = torch.unsqueeze(
        sequence_mask(refer_lengths, refer_spec.size(2)), 1
    ).to(refer_spec.dtype)
    print(refer_spec.shape)
    ge = tts.vits_model.ref_enc(refer_spec[:, :704] * refer_mask, refer_mask)
    print(ge.shape)
    phones_batch, bert_features_batch, norm_text_batch, _ = (
        tts.text_preprocessor.segment_and_extract_feature_for_text(
            prompt_texts_batch, prompt_lang, is_prompt=True
        )
    )
    voice_data = {
        "prompt_semantic": prompt_semantic[0].cpu(),
        "ge": ge.cpu(),
        "phones": phones_batch[0],
        "bert_features": bert_features_batch[0].cpu(),
        "norm_text": norm_text_batch[0],
        "prompt_text": prompt_text,
    }
    
    # 将数据转换为pickle格式
    pickled_data = pickle.dumps(voice_data)
    compressed_data = zlib.compress(pickled_data)
    # 打印compressed_data的大小，用MB表示
    compressed_size_mb = len(compressed_data) / (1024 * 1024)
    print(f"压缩后的数据大小: {compressed_size_mb:.2f} MB")
    # 准备存储到MongoDB的文档
    document = {
        "_id": id,
        "voice_data": Binary(compressed_data),  # 使用Binary包装pickle数据
        "length": refer_spec.shape[2]/50,
    }


    # 存储到MongoDB
    collection.replace_one({"_id": id}, document, upsert=True)
    # 从MongoDB读取数据


    document = collection.find_one({"_id": id})
    if document:
        voice_data = pickle.loads(zlib.decompress(document["voice_data"]))
        if torch.cuda.is_available():
            voice_data["prompt_semantic"] = voice_data["prompt_semantic"].cuda()
            voice_data["ge"] = voice_data["ge"].cuda()
            voice_data["bert_features"] = voice_data["bert_features"].cuda()
        print(voice_data)
        
    return {"message": "Voice created and stored successfully", "id": id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)