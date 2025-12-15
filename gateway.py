import asyncio
import io
import os
import pickle
import re
import traceback
import uuid
import zlib
from typing import List

import aiocache
import aiohttp
import motor.motor_asyncio as motor
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketState
from pydantic import BaseModel
import numpy as np
from pydub import AudioSegment

load_dotenv(dotenv_path="./.env")
#load_dotenv(dotenv_path="/workspaces/.env")

# MongoDB 连接配置
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

app = FastAPI()
# 请求队列
short_request_queue = asyncio.Queue()  # 用于150-250长度的请求
long_request_queue = asyncio.Queue()  # 用于251-400长度的请求
client_session = None
# 机器（IP地址）池
# 存储请求状态和结果的字典
request_status = {}
CONNECTION_STRING = f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/voice?authSource=admin"

client = motor.AsyncIOMotorClient(
    CONNECTION_STRING,
    maxPoolSize=50,  # 最大连接数
    minPoolSize=5,  # 最小连接数
    maxIdleTimeMS=30000,  # 连接空闲时间（毫秒���
    waitQueueMultiple=10,  # 等待队列的大小
    waitQueueTimeoutMS=10000,  # 等待队列超时时间（毫秒）
)
database = client.voice.voice_cache_test
machine_connections = {}
# 创建一个锁
websocket_locks = {}


class MachineModel(BaseModel):
    ip: str


@aiocache.cached(ttl=604800)
async def get_length(id: str):
    return (await database.find_one({"_id": id}, projection={"length": 1}))["length"]


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(handle_queue())


@app.on_event("shutdown")
async def shutdown_event():
    for connection in machine_connections.values():
        await connection.close()


async def process_request(batch_data, batch_requests, queue, websocket, message_queue):
    lock = websocket_locks.get(websocket)
    if not lock:
        lock = asyncio.Lock()
        websocket_locks[websocket] = lock

    async with lock:
        try:
            await websocket.send_json(batch_data)

            # 从消息队列中获取机器的响应
            response_data = await asyncio.wait_for(message_queue.get(), timeout=80.0)
            audios = zlib.decompress(response_data["bytes"])
            audios = pickle.loads(audios)
            audios = audios[1:]
            
            # 添加安全检查
            if len(audios) != len(batch_requests):
                raise Exception(f"音频数量不匹配: 期望 {len(batch_requests)}, 实际 {len(audios)}")

            # 处理结果
            for i, (request_id, _, event) in enumerate(batch_requests):
                request_status[request_id]["result"] = audios[i]
                event.set()
                queue.task_done()

        except Exception as e:
            error_message = f"处理失败: {str(e)}"
            print(traceback.format_exc())
            for request_id, _, event in batch_requests:
                request_status[request_id]["result"] = error_message
                event.set()
                queue.task_done()

        finally:
            # 只有在 WebSocket 连接仍然打开的情况下，才将其重新添加到 machine_connections 中
            if not websocket.client_state == WebSocketState.DISCONNECTED:
                machine_connections[websocket] = message_queue
            else:
                print(
                    f"WebSocket connection {websocket} is closed, not adding back to machine_connections"
                )


async def handle_queue():
    while True:
        for queue in [short_request_queue, long_request_queue]:
            if not queue.empty() and machine_connections:
                websocket, message_queue = next(iter(machine_connections.items()))
                del machine_connections[websocket]

                batch_size = 48
                batch_requests = []

                while len(batch_requests) < batch_size:
                    try:
                        request = queue.get_nowait()
                        batch_requests.append(request)
                    except asyncio.QueueEmpty:
                        break

                request_id = str(uuid.uuid4())
                batch_data = {
                    "request_id": request_id,
                    "texts": [req[1]["text"] for req in batch_requests],
                    "all_ids": [req[1]["ids"] for req in batch_requests],
                }
                print(f"一次性处理{len(batch_requests)}个请求")
                asyncio.create_task(
                    process_request(
                        batch_data, batch_requests, queue, websocket, message_queue
                    )
                )

        await asyncio.sleep(0.01)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    目标机器通过此接口连接到gateway，连接后将WebSocket加入连接池。
    """
    await websocket.accept()
    message_queue = asyncio.Queue()  # 为每个WebSocket连接创建一个消息队列
    machine_connections[websocket] = message_queue
    try:
        while True:
            # 保持连接，等待机器发送消息，并将消息放入队列
            message = await websocket.receive()
            await message_queue.put(message)  # 将消息放入队列
    except Exception:
        print(traceback.format_exc())
    finally:
        # 连接关闭时从池子中移除
        del machine_connections[websocket]
        await websocket.close()


class TTSRequest(BaseModel):
    text: str
    ids: List[str]

def replace_words(text: str):
    text = "." + text
    text = re.sub(r"\.{6}", ",", text)
    text = re.sub(r"\.{3}", ",", text)
    rule = {
        "Rubii": "Luby",
        "rubii": "luby",
    }
    for key, value in rule.items():
        text = text.replace(key, value)
    return text


@app.post("/tts")
async def text2speech(form_data: TTSRequest):
    text = form_data.text
    text = replace_words(text)
    ids = form_data.ids
    length = await get_length(ids[0])

    # 创建一个包含文本和语音数据的字典
    request_data = {
        "text": text,
        "ids": ids,
    }

    request_id = str(uuid.uuid4())

    # 创建一个事件对象，用于等待请求处理完成
    event = asyncio.Event()

    # 将请求添加到状态字典中
    request_status[request_id] = {"status": "pending", "result": None}

    # 根据prompt_semantic的长度选择队列
    if 1 <= length <= 5:
        await short_request_queue.put((request_id, request_data, event))
    elif 5 < length <= 20:
        await long_request_queue.put((request_id, request_data, event))
    else:
        print(f"Invalid length: {length}, id: {ids[0]}")
        raise HTTPException(status_code=400, detail="Invalid length")

    # 等待请求处理完成
    await event.wait()
    # 获取处理结果
    result = request_status[request_id]["result"]
    if isinstance(result, str):
        raise HTTPException(status_code=500, detail=result)
    # 清理状态字典
    del request_status[request_id]


    # 返回MP3格式的音频
    return StreamingResponse(io.BytesIO(result), media_type="audio/mpeg")


@app.post("/create_voice")
async def create_voice(id: str, file: UploadFile = File(...)):
    try:
        data = aiohttp.FormData()
        data.add_field(
            "file",
            await file.read(),
            filename=file.filename,
            content_type=file.content_type,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:3000/create_voice?id=" + id, data=data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(
                        status_code=response.status, detail="Failed to create voice"
                    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating voice: {str(e)}")


@app.get("/queue_status")
async def queue_status():
    available_machines = list(machine_connections)
    return {
        "queue_size": short_request_queue.qsize() + long_request_queue.qsize(),
        "pending_requests": len(request_status),
        "available_machines": available_machines,
        "num_machines": len(machine_connections),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000, loop="uvloop")
