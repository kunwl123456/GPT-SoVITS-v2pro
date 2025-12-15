# ruff: noqa
import asyncio
import json
import logging
import os
import pickle
import sys
import zlib
import gc
import torch
import time
from dotenv import load_dotenv

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/GPT_SoVITS" % (now_dir))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import motor.motor_asyncio as motor
from aiocache import cached
from pydantic import BaseModel
import argparse
import traceback
import websockets
import yaml

from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import (
    get_method_names as get_cut_method_names,
)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from tools.i18n.i18n import I18nAuto

# 设置pymongo的日志级别为WARNING，这样就不会输出DEBUG信息
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("pydub").setLevel(logging.WARNING)

load_dotenv(dotenv_path="./.env")

# MongoDB 连接配置
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

i18n = I18nAuto()
cut_method_names = get_cut_method_names()


argv = sys.argv
config_path = os.path.join(now_dir, "GPT_SoVITS", "configs", "tts_infer.yaml")
print(f"[create_voice] 配置文件路径: {config_path}")
print(f"[create_voice] 配置文件存在: {os.path.exists(config_path)}")
with open(config_path, "r", encoding="utf-8") as f:
    cfg_loaded = yaml.load(f, Loader=yaml.FullLoader)
print(f"[create_voice] yaml.load 类型: {type(cfg_loaded)}")
tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)

CONNECTION_STRING = f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/voice?authSource=admin"

client = motor.AsyncIOMotorClient(
    CONNECTION_STRING,
    maxPoolSize=50,  # 最大连接数
    minPoolSize=5,  # 最小连接数
    maxIdleTimeMS=30000,  # 连接空闲时间（毫秒）
    waitQueueMultiple=10,  # 等待队列的大小
    waitQueueTimeoutMS=10000,  # 等待队列超时时间（毫秒）
)
database = client.voice.voice_cache_test
@cached(ttl=604800)
async def get_prompt_cache_by_id(id: str):
    # 从 MongoDB 查询数据
    document = await database.find_one({"_id": id})
    if document:
        return pickle.loads(zlib.decompress(document["voice_data"]))
    else:
        raise Exception("Prompt not found")


async def prepare_prompt_caches(body):
    prompt_caches = []
    all_ids = []
    all_ids = sum(body["all_ids"], [])

    # 使用 asyncio.gather 一次性获取所有 id 对应的缓存
    cache_results = await asyncio.gather(
        *[get_prompt_cache_by_id(id) for id in all_ids]
    )

    prompt_cache_dict = {id: result for id, result in zip(all_ids, cache_results)}

    for i, ids in enumerate(body["all_ids"]):
        prompt_cache = {
            "id": i,
            "prompt_semantic": prompt_cache_dict[ids[0]]["prompt_semantic"],
            "ge": sum((prompt_cache_dict[id]["ge"] for id in ids)) / len(ids),
            "phones": prompt_cache_dict[ids[0]]["phones"],
            "norm_text": prompt_cache_dict[ids[0]]["norm_text"],
            "prompt_text": prompt_cache_dict[ids[0]]["prompt_text"],
            "bert_features": prompt_cache_dict[ids[0]]["bert_features"],
        }
        prompt_caches.append(prompt_cache)
    body["prompt_caches"] = prompt_caches
    return body


async def service(gateway_url):
    global tts_pipeline
    while True:
        try:
            async with websockets.connect(gateway_url) as websocket:
                print("成功连接到 gateway")
                while True:
                    try:
                        # 接收来自 gateway 的请求
                        data = await websocket.recv()
                        print("收到请求:", data)
                        body = json.loads(data)
                        request_id = body["request_id"]

                        # 处理请求
                        body = await prepare_prompt_caches(body)
                        audios = await tts_pipeline.run_with_cache(body, websocket, body["request_id"])

                        # 将 audios 转换为 pickle 格式并压缩
                        audios.insert(0, request_id)
                        pickled_audios = pickle.dumps(audios)
                        compressed_audios = zlib.compress(pickled_audios)

                        # 发送处理结果回 gateway
                        await websocket.send(compressed_audios)
                        print(f"已发送音频文件给: {request_id}")

                    except Exception as e:
                        print(f"处理请求时出错: {e}")
                        print(traceback.format_exc())
                        await websocket.send(json.dumps({
                            "status": "error",
                            "request_id": request_id,
                            "message": str(e)
                        }))
                        if str(e).lower().find("cuda") != -1:
                            try:
                                tts_pipeline.clean_up()
                                del tts_pipeline
                                gc.collect()
                                torch.cuda.empty_cache()
                                time.sleep(1)
                                tts_pipeline=TTS(tts_config)
                                continue
                            except Exception:
                                print(traceback.format_exc())
                                exit(0)
        except Exception as e:
            print(f"无法连接到 gateway: {e}")
            await asyncio.sleep(1)  # 连接失败后等待一段时间再重试


if __name__ == "__main__":
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Service to connect to Gateway")
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="ws://localhost:7000/ws",
        help="The WebSocket URL of the gateway to connect to"
    )
    args = parser.parse_args()

    # 获取传入的 gateway URL
    gateway_url = args.gateway_url

    # 启动时主动连接到 gateway
    loop = asyncio.get_event_loop()
    loop.run_until_complete(service(gateway_url))
