#!/usr/bin/env python3
"""
Ollama 连接测试脚本

使用方法:
    python test_ollama.py

确保 Ollama 服务已启动:
    1. 安装 Ollama: https://ollama.com/
    2. 启动服务: ollama serve
    3. 下载模型: ollama pull qwen2.5:7b
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from llm_providers import OllamaProvider

# 加载 .env 配置
load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')


def test_ollama_connection():
    """测试 Ollama 连接和基本功能"""
    print("=" * 60)
    print("Ollama 连接测试")
    print("=" * 60)
    print(f"服务地址: {OLLAMA_BASE_URL}")
    print(f"模型名称: {OLLAMA_MODEL}")
    print()

    # 创建 provider 实例
    provider = OllamaProvider(OLLAMA_BASE_URL, OLLAMA_MODEL)

    # 测试连接
    print("步骤 1: 测试连接...")
    try:
        client = provider.configure()
        if not client:
            print("连接失败！请检查:")
            print("  1. Ollama 服务是否正在运行 (执行: ollama serve)")
            print("  2. 模型是否已下载 (执行: ollama pull qwen2.5:7b)")
            print("  3. .env 配置是否正确")
            return False
        print("连接成功！")
    except Exception as e:
        print(f"连接失败: {e}")
        return False

    # 列出可用模型
    print("\n步骤 2: 获取模型列表...")
    try:
        models = provider.list_models()
        print(f"本地可用模型 ({len(models)} 个):")
        for model in models:
            model_name = model.get('name', 'Unknown')
            size = model.get('size', 0) / (1024**3) if model.get('size') else 0
            print(f"  - {model_name} ({size:.1f} GB)")
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        # 这不是致命错误，继续测试

    # 测试简单对话
    print(f"\n步骤 3: 测试 {OLLAMA_MODEL} 对话能力...")
    test_prompt = "请用一句话介绍你自己，包括你的模型名称和擅长领域。"
    try:
        print(f"输入: {test_prompt}")
        response = provider.generate_content(test_prompt)
        print(f"回复: {response}")
    except Exception as e:
        print(f"对话测试失败: {e}")
        return False

    # 测试歌词清理提示词
    print("\n步骤 4: 测试歌词清理提示词...")
    test_lrc = """[00:12.34]这是第一句歌词
[00:15.67](间奏音乐)
[00:18.90]这是第二句歌词
[00:22.11]...省略号歌词
[00:25.32]这是第三句歌词
[00:28.55]纯音乐部分
[00:31.78]这是第四句歌词"""

    clean_prompt = f"""
    你是一个专业的歌词内容审查员。删除以下 LRC 内容中不属于歌词的部分，只保留歌词本身。
    规则：
    1. 只删除包含 "(音乐)", "...", "纯音乐", "间奏" 等非歌词内容
    2. 保留所有时间戳和歌词
    3. 不要修改任何歌词内容

    待处理内容:
    ---
    {test_lrc}
    ---
    """
    try:
        print("输入歌词清理测试...")
        response = provider.generate_content(clean_prompt)
        print("清理结果:")
        print(response)
    except Exception as e:
        print(f"歌词清理测试失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    return True


def main():
    """主函数"""
    success = test_ollama_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
