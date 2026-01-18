"""
Google Gemini LLM 提供者实现
"""

import google.generativeai as genai
from typing import Optional
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API 提供者"""

    def __init__(self, api_key: str, model_name: str):
        """
        初始化 Gemini 提供者

        Args:
            api_key: Google Gemini API 密钥
            model_name: Gemini 模型名称，如 'gemini-2.5-flash'
        """
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def configure(self) -> Optional[object]:
        """
        配置 Gemini 模型

        Returns:
            GenerativeModel 实例，配置失败返回 None
        """
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY":
            return None
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            return self._model
        except Exception as e:
            print(f"配置 Gemini 模型失败: {e}")
            return None

    def generate_content(self, prompt: str) -> str:
        """
        使用 Gemini 生成内容

        Args:
            prompt: 输入的提示词

        Returns:
            生成的文本内容

        Raises:
            RuntimeError: 如果模型未配置
        """
        if not self._model:
            raise RuntimeError("Gemini 模型未配置，请先调用 configure()")
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def provider_name(self) -> str:
        """返回提供者名称"""
        return "Gemini"
