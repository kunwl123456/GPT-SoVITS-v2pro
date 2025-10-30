#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多语言TTS测试脚本
为每个已创建的角色生成日文、中文、英文、韩语四种语言的测试音频
"""

import os
import requests
import time
from pathlib import Path

# 配置
GATEWAY_URL = "http://localhost:8000"
TTS_API = f"{GATEWAY_URL}/tts"
OUTPUT_DIR = "./test_output"

# 测试文本（四种语言）
TEST_TEXTS = {
    "zh": "亲爱的我想要了，你快给我，狠狠的爱我吧",
    "ja": "こんにちは、音声合成テストです。今日はいい天気ですね、一緒に散歩しましょう。",
    "en": "Hello, this is a text-to-speech test. The weather is nice today, let's go for a walk together.",
    "ko": "안녕하세요, 음성 합성 테스트입니다. 오늘 날씨가 정말 좋네요, 함께 산책하러 가요."
}

def get_all_voice_ids():
    """
    从 divice_result 目录获取所有已创建的角色ID
    """
    base_dir = "./divice_result"
    voice_ids = []
    
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return voice_ids
    
    for category_dir in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_dir)
        
        if not os.path.isdir(category_path):
            continue
        
        for character_folder in os.listdir(category_path):
            character_path = os.path.join(category_path, character_folder)
            
            if not os.path.isdir(character_path):
                continue
            
            # 提取角色名（与 batch_create_voice.py 逻辑一致）
            if "]" in character_folder:
                character_name = character_folder.split("]")[1].strip()
            else:
                character_name = character_folder.strip()
            
            voice_ids.append(character_name)
    
    return voice_ids

def generate_audio(voice_id, text, lang):
    """
    调用 TTS API 生成音频，等待完成
    """
    try:
        payload = {
            "text": text,
            "ids": [voice_id]
        }
        
        print(f"  正在生成 [{lang}]: {voice_id}")
        
        # 发送请求并等待响应（最多等待 2 分钟）
        response = requests.post(TTS_API, json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.content  # 返回音频字节数据
        else:
            print(f"    ✗ 失败: HTTP {response.status_code}")
            print(f"    错误: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"    ✗ 超时: 请求超过 2 分钟")
        return None
    except Exception as e:
        print(f"    ✗ 错误: {str(e)}")
        return None

def save_audio(audio_data, voice_id, lang):
    """
    保存音频文件到本地
    """
    # 创建输出目录
    voice_dir = os.path.join(OUTPUT_DIR, voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    
    # 保存文件
    output_file = os.path.join(voice_dir, f"{voice_id}_{lang}.mp3")
    with open(output_file, 'wb') as f:
        f.write(audio_data)
    
    return output_file

def test_voice(voice_id):
    """
    为单个角色生成四种语言的测试音频
    """
    print(f"\n{'='*60}")
    print(f"测试角色: {voice_id}")
    print(f"{'='*60}")
    
    success_count = 0
    
    for lang, text in TEST_TEXTS.items():
        # 生成音频（同步等待）
        audio_data = generate_audio(voice_id, text, lang)
        
        if audio_data:
            # 保存音频
            output_file = save_audio(audio_data, voice_id, lang)
            file_size = len(audio_data) / 1024  # KB
            print(f"    ✓ 成功: {output_file} ({file_size:.1f} KB)")
            success_count += 1
            
            # 等待 1 秒再生成下一个语言（避免请求过快）
            time.sleep(1)
        else:
            print(f"    ✗ 跳过: {lang}")
    
    return success_count

def retry_failed_voices(specific_voices=None):
    """
    重新为失败的角色生成音频
    
    Args:
        specific_voices: 指定要重试的角色列表，如果为 None 则自动检测失败的角色
    """
    print("="*60)
    print("重新生成失败角色的音频")
    print("="*60)
    
    languages = ["zh", "ja", "en", "ko"]
    
    if specific_voices is None:
        # 自动检测失败的角色
        all_voices = get_all_voice_ids()
        failed_voices = []
        
        for voice_id in all_voices:
            voice_dir = os.path.join(OUTPUT_DIR, voice_id)
            
            # 检查是否完全失败（目录不存在）
            if not os.path.exists(voice_dir):
                failed_voices.append(voice_id)
                continue
            
            # 检查是否部分失败（缺少某些语言）
            for lang in languages:
                audio_file = os.path.join(voice_dir, f"{voice_id}_{lang}.mp3")
                if not os.path.exists(audio_file):
                    if voice_id not in failed_voices:
                        failed_voices.append(voice_id)
                    break
        
        if not failed_voices:
            print("\n没有发现失败的角色，所有音频都已生成！")
            return
        
        print(f"\n发现 {len(failed_voices)} 个需要重试的角色:")
        for vid in failed_voices:
            print(f"  - {vid}")
    else:
        failed_voices = specific_voices
        print(f"\n指定重试 {len(failed_voices)} 个角色:")
        for vid in failed_voices:
            print(f"  - {vid}")
    
    # 重新生成
    total_success = 0
    total_fail = 0
    
    for i, voice_id in enumerate(failed_voices, 1):
        print(f"\n[{i}/{len(failed_voices)}] 重试角色: {voice_id}")
        
        success = test_voice(voice_id)
        total_success += success
        total_fail += (4 - success)
        
        # 完成一个角色后等待 3 秒
        if i < len(failed_voices):
            print(f"\n  等待 3 秒后继续...")
            time.sleep(3)
    
    # 统计
    print(f"\n{'='*60}")
    print(f"重试完成!")
    print(f"重试角色数: {len(failed_voices)}")
    print(f"成功生成: {total_success} 个音频")
    print(f"失败: {total_fail} 个音频")
    print(f"{'='*60}")

def main():
    """
    主函数：为所有角色生成多语言测试音频
    """
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--retry":
            # 重试模式
            if len(sys.argv) > 2:
                # 指定角色列表
                specific_voices = sys.argv[2:]
                retry_failed_voices(specific_voices)
            else:
                # 自动检测失败角色
                retry_failed_voices()
            return
        elif sys.argv[1] == "--help":
            print("用法:")
            print("  python test_multilang_tts.py              # 为所有角色生成音频")
            print("  python test_multilang_tts.py --retry      # 自动重试失败的角色")
            print("  python test_multilang_tts.py --retry 夜神 娜莱妮  # 重试指定角色")
            return
    
    print("="*60)
    print("多语言TTS测试脚本")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有角色ID
    voice_ids = get_all_voice_ids()
    
    if not voice_ids:
        print("错误: 没有找到任何角色")
        return
    
    print(f"\n找到 {len(voice_ids)} 个角色:")
    for vid in voice_ids:
        print(f"  - {vid}")
    
    # 为每个角色生成测试音频
    total_success = 0
    total_fail = 0
    
    for i, voice_id in enumerate(voice_ids, 1):
        print(f"\n[{i}/{len(voice_ids)}] 处理角色: {voice_id}")
        
        success = test_voice(voice_id)
        total_success += success
        total_fail += (4 - success)  # 每个角色应该生成4个音频
        
        # 完成一个角色后等待 3 秒再处理下一个
        if i < len(voice_ids):
            print(f"\n  等待 3 秒后处理下一个角色...")
            time.sleep(3)
    
    # 统计
    print(f"\n{'='*60}")
    print(f"测试完成!")
    print(f"总角色数: {len(voice_ids)}")
    print(f"成功生成: {total_success} 个音频")
    print(f"失败: {total_fail} 个音频")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

