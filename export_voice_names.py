#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出语音文件名脚本
根据 divice_result 目录下的音频文件，提取角色名并导出到 txt 文件
"""

import os
import re
from pathlib import Path

# 配置
BASE_DIR = "./divice_result"
OUTPUT_FILE = "voice_names_list.txt"

def extract_character_name(filename):
    """
    从文件名提取角色名
    例如: "teen_male_Abedo.wav" -> "Abedo"
          "child_female_Qiqi.wav" -> "Qiqi"
    """
    # 去除 .wav 后缀
    name = filename.replace('.wav', '')
    # 按下划线分割，取第三部分开始的所有内容
    parts = name.split('_', 2)  # 最多分割2次
    if len(parts) >= 3:
        return parts[2]  # 返回角色名部分
    return name

def clean_voice_id(character_name):
    """
    清理角色名，生成纯英文ID（去除空格、标点符号等）
    例如: "Danheng • Drinking Moon" -> "DanhengDrinkingMoon"
          "March 7th!" -> "March7th"
          "O'Loren" -> "OLoren"
    """
    # 移除所有非字母数字字符（保留字母和数字）
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', character_name)
    return cleaned

def create_voice_id(category, character):
    """
    生成语音ID
    例如: category="teen_male", character="Danheng • Drinking Moon" -> "DanhengDrinkingMoon"
    """
    return clean_voice_id(character)

def main():
    """
    主函数：遍历所有分类目录下的音频文件，导出角色名列表
    """
    if not os.path.exists(BASE_DIR):
        print(f"错误: 目录 {BASE_DIR} 不存在")
        return
    
    # 用于存储所有结果
    all_results = []
    total_count = 0
    
    # 遍历所有分类目录 (teen_male, teen_female, child_male, child_female, etc.)
    for category_dir in sorted(os.listdir(BASE_DIR)):
        category_path = os.path.join(BASE_DIR, category_dir)
        
        if not os.path.isdir(category_path):
            continue
        
        print(f"\n{'='*60}")
        print(f"处理分类: {category_dir}")
        print(f"{'='*60}")
        
        # 遍历该分类下的所有 .wav 文件
        wav_files = sorted([f for f in os.listdir(category_path) if f.endswith('.wav')])
        
        if not wav_files:
            print(f"⚠ 跳过: {category_dir} (没有找到 .wav 文件)")
            continue
        
        print(f"找到 {len(wav_files)} 个音频文件\n")
        
        # 分类结果
        category_results = []
        category_results.append(f"\n{'='*60}")
        category_results.append(f"分类: {category_dir}")
        category_results.append(f"{'='*60}\n")
        
        # 处理每个音频文件
        for wav_file in wav_files:
            # 从文件名提取角色名
            character_name = extract_character_name(wav_file)
            
            # 生成纯英文 voice_id（无空格无标点）
            voice_id = create_voice_id(category_dir, character_name)
            
            # 格式化输出行
            line = f"{voice_id:30s} | 原名: {character_name:35s} | 文件: {wav_file}"
            category_results.append(line)
            
            # 显示到控制台
            print(f"  {voice_id:25s} <- {character_name}")
            
            total_count += 1
        
        # 添加分类结果到总结果
        all_results.extend(category_results)
        all_results.append("")  # 空行分隔
    
    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 写入头部
        f.write("="*80 + "\n")
        f.write("语音角色名称列表\n")
        f.write(f"总计: {total_count} 个语音文件\n")
        f.write("="*80 + "\n")
        
        # 写入所有结果
        for line in all_results:
            f.write(line + "\n")
        
        # 写入尾部统计
        f.write("\n" + "="*80 + "\n")
        f.write(f"导出完成! 共 {total_count} 个语音角色\n")
        f.write("="*80 + "\n")
    
    # 统计
    print(f"\n{'='*60}")
    print(f"导出完成!")
    print(f"共处理: {total_count} 个语音文件")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()




