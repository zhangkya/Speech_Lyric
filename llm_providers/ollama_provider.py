"""
Ollama LLM 提供者实现
支持本地运行的 Ollama 模型
"""

import ollama
from typing import Optional
from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama 本地 LLM 提供者"""

    def __init__(self, base_url: str, model_name: str):
        """
        初始化 Ollama 提供者

        Args:
            base_url: Ollama 服务地址，默认 'http://localhost:11434'
            model_name: Ollama 模型名称，如 'qwen2.5:7b', 'llama3:8b'
        """
        self.base_url = base_url
        self.model_name = model_name
        self._client = None

    def configure(self) -> Optional[object]:
        """
        配置 Ollama 客户端

        Returns:
            Ollama Client 实例，配置失败返回 None
        """
        try:
            # 创建 Ollama 客户端（通过 host 参数指定服务地址）
            self._client = ollama.Client(host=self.base_url)

            # 测试连接
            self._client.list()
            return self._client
        except AttributeError:
            # 旧版本 ollama 可能不支持 host 参数，尝试直接使用 ollama 模块
            try:
                import ollama as ollama_module
                # 测试连接
                ollama_module.list()
                self._client = ollama_module
                return self._client
            except Exception as e2:
                print(f"配置 Ollama 失败: {e2}")
                print(f"   服务地址: {self.base_url}")
                print(f"   请确保 Ollama 服务正在运行（执行: ollama serve）")
                print(f"   如需安装/更新 ollama Python 包，请运行: pip install -U ollama")
                return None
        except Exception as e:
            print(f"配置 Ollama 失败（请确保 Ollama 服务正在运行）: {e}")
            print(f"   服务地址: {self.base_url}")
            print(f"   如需启动 Ollama，请访问: https://ollama.com/")
            return None

    def generate_content(self, prompt: str) -> str:
        """
        使用 Ollama 生成内容

        Args:
            prompt: 输入的提示词

        Returns:
            生成的文本内容

        Raises:
            RuntimeError: 如果客户端未配置
        """
        if not self._client:
            raise RuntimeError("Ollama 客户端未配置，请先调用 configure()")

        try:
            # 尝试新版本 API (Client.chat)
            if hasattr(self._client, 'chat'):
                response = self._client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False
                )
                return response['message']['content']
            # 尝试旧版本 API (ollama.generate)
            else:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False
                )
                return response.get('response', '')
        except Exception as e:
            raise RuntimeError(f"Ollama 生成内容失败: {e}")

    def list_models(self) -> list:
        """
        列出所有可用的 Ollama 模型

        Returns:
            模型列表，每个模型包含 'name' 和其他元信息
        """
        if not self._client:
            raise RuntimeError("Ollama 客户端未配置，请先调用 configure()")

        try:
            models_info = self._client.list()
            return models_info.get('models', [])
        except Exception as e:
            raise RuntimeError(f"获取 Ollama 模型列表失败: {e}")

    @property
    def provider_name(self) -> str:
        """返回提供者名称"""
        return "Ollama"
