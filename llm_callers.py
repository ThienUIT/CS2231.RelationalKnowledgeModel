import os
import time
from typing import Optional
from dotenv import load_dotenv

import google.generativeai as genai
import openai
import ollama
from openai import OpenAI

load_dotenv()


# --------------------------------
# Base Provider Interface
# --------------------------------
class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# --------------------------------
# Gemini Provider
# --------------------------------
class GeminiProvider(BaseLLM):

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured")

        genai.configure(api_key=api_key)

        models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        priority = [
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]

        selected = None
        for p in priority:
            if p in models:
                selected = p
                break

        if not selected:
            selected = models[0]

        print(f"[LLM] Gemini using: {selected}")

        self.model = genai.GenerativeModel(selected)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


# --------------------------------
# OpenAI Provider
# --------------------------------
class OpenAIProvider(BaseLLM):

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message["content"].strip()


# --------------------------------
# DeepSeek Provider
# --------------------------------
class DeepSeekProvider(BaseLLM):

    def __init__(self, model: str = "deepseek-chat"):

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not configured")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        self.model = model

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content


# --------------------------------
# Ollama Provider (Local Default)
# --------------------------------
class OllamaProvider(BaseLLM):

    def __init__(
        self,
        model: str = "minimax-m2.5:cloud",
    ):
        self.model = model

    def _load_prompt(self, path: str) -> str:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def generate(self, prompt: str) -> str:

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia pháp lý cao cấp về Luật Đất Đai Việt Nam"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

class LLMClient:

    def __init__(self, default_provider: str = "ollama"):

        self.default_provider = default_provider

    def _get_provider(self, provider: str) -> BaseLLM:

        if provider == "ollama":
            return OllamaProvider()

        if provider == "gemini":
            return GeminiProvider()

        if provider == "openai":
            return OpenAIProvider()

        if provider == "deepseek":
            return DeepSeekProvider()

        raise ValueError(f"Unknown provider: {provider}")

    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        retries: int = 3
    ) -> str:

        provider = provider or self.default_provider
        llm = self._get_provider(provider)

        for attempt in range(retries):

            try:
                return llm.generate(prompt)

            except Exception as e:

                print(f"[LLM] {provider} error {attempt+1}/{retries}: {e}")

                if attempt < retries - 1:
                    time.sleep(3)

        raise RuntimeError(f"{provider} failed after retries")


if __name__ == "__main__":

    llm = LLMClient(default_provider="gemini")

    prompt = "Luật đất đai 2024 có hiệu lực khi nào?"

    print("\n--- Local Ollama ---")
    print(llm.generate(prompt))

    print("\n--- Gemini ---")
    print(llm.generate(prompt, "gemini"))

    print("\n--- OpenAI ---")
    print(llm.generate(prompt, "openai"))

    print("\n--- DeepSeek ---")
    print(llm.generate(prompt, "deepseek"))