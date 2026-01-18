"""
LLM 提供者基础接口
定义所有 LLM 提供者必须实现的抽象方法
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """LLM 提供者基础接口"""

    @abstractmethod
    def configure(self) -> Optional[object]:
        """
        配置并返回模型实例

        Returns:
            配置好的模型实例，如果配置失败则返回 None
        """
        pass

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """
        生成内容

        Args:
            prompt: 输入的提示词

        Returns:
            生成的文本内容
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        提供者名称

        Returns:
            提供者的名称，如 'Gemini', 'Ollama' 等
        """
        pass
