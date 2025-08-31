#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import configure_llm, get_song_background_story

def main():
    # 配置LLM模型
    llm_model = configure_llm()
    
    if not llm_model:
        print("错误: 无法配置LLM模型。请检查.env文件中的配置。")
        return
    
    # 测试歌曲背景故事功能
    artist = "BENI"
    title = "LA·LA·LA LOVE SONG"
    
    print(f"正在搜索歌曲'{title}'的背景故事...")
    story = get_song_background_story(llm_model, artist, title)
    
    print("\n歌曲背景故事:")
    print("=" * 50)
    print(story)
    print("=" * 50)

if __name__ == "__main__":
    main()
