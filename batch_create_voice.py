#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量创建语音数据脚本
根据 divice_result 目录下的音频文件，自动调用 create_voice API 创建语音缓存
"""

import os
import requests
from pathlib import Path
import time

# 配置
API_URL = "http://localhost:4000/create_voice"
BASE_DIR = "./divice_result"

def extract_character_name(folder_name):
    """
    从文件夹名提取角色名
    例如: "[萝莉] 七七" -> "七七"
    """
    if "]" in folder_name:
        return folder_name.split("]")[1].strip()
    return folder_name.strip()

def create_voice_id(category, character):
    """
    生成语音ID
    例如: category="萝莉", character="七七" -> "七七"
    """
    return character

def upload_audio(voice_id, audio_path):
    """
    上传音频文件到 create_voice API，等待处理完成
    """
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
            data = {'id': voice_id}
            
            print(f"正在上传: {voice_id} <- {os.path.basename(audio_path)}")
            
            # 设置较长的超时时间，等待服务器完成语音识别和特征提取
            response = requests.post(API_URL, files=files, data=data, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 成功: {voice_id}")
                if 'message' in result:
                    print(f"  响应: {result['message']}")
                return True
            else:
                print(f"✗ 失败: {voice_id} - HTTP {response.status_code}")
                print(f"  错误: {response.text[:200]}")
                return False
    except requests.exceptions.Timeout:
        print(f"✗ 超时: {voice_id} - 请求超过 10 分钟")
        return False
    except Exception as e:
        print(f"✗ 错误: {voice_id} - {str(e)}")
        return False

def main():
    """
    主函数：遍历所有分类和角色，批量创建语音数据
    """
    if not os.path.exists(BASE_DIR):
        print(f"错误: 目录 {BASE_DIR} 不存在")
        return
    
    success_count = 0
    fail_count = 0
    
    # 遍历所有分类目录（萝莉、正太、成男、成女、少年、少女）
    for category_dir in os.listdir(BASE_DIR):
        category_path = os.path.join(BASE_DIR, category_dir)
        
        if not os.path.isdir(category_path):
            continue
        
        print(f"\n{'='*60}")
        print(f"处理分类: {category_dir}")
        print(f"{'='*60}")
        
        # 遍历该分类下的所有角色目录
        for character_folder in os.listdir(category_path):
            character_path = os.path.join(category_path, character_folder)
            
            if not os.path.isdir(character_path):
                continue
            
            # 提取角色名
            character_name = extract_character_name(character_folder)
            
            # 查找该角色目录下的第一个 .wav 文件
            wav_files = [f for f in os.listdir(character_path) if f.endswith('.wav')]
            
            if not wav_files:
                print(f"⚠ 跳过: {character_folder} (没有找到 .wav 文件)")
                continue
            
            # 使用第一个 wav 文件
            audio_file = os.path.join(character_path, wav_files[0])
            
            # 生成 voice_id
            voice_id = create_voice_id(category_dir, character_name)
            
            # 上传并等待完成
            if upload_audio(voice_id, audio_file):
                success_count += 1
                print(f"  等待 3 秒后继续...")
                time.sleep(3)  # 等待上一个角色处理完成
            else:
                fail_count += 1
                time.sleep(1)  # 失败也稍作等待
    
    # 统计
    print(f"\n{'='*60}")
    print(f"批量创建完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

