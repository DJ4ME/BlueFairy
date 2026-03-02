import os
import openai
from core import LanguageModelProvider, LanguageModel


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

class OpenAIService(LanguageModelProvider):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def use(self, language_model: str, system_prompt: str) -> LanguageModel:
        return OpenAILanguageModel(
            model_name=language_model,
            system_prompt=system_prompt,
        )

    def __str__(self):
        return f"OpenAIService(model={self.model})"

    def __repr__(self):
        return self.__str__()

class OpenAILanguageModel(LanguageModel):
    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt

        openai.api_key = OPENAI_API_KEY
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

    def ask(
        self,
        question: str | list[str],
        max_output: int = 1024,
        temperature: float = 0.0,
    ) -> str | list[str]:

        if isinstance(question, str):
            questions = [question]
            single = True
        else:
            questions = question
            single = False

        responses = []
        for q in questions:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": q})
            resp = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_output,
                temperature=temperature,
            )
            answer = resp.choices[0].message.content
            responses.append(answer)
        return responses[0] if single else responses

    def __str__(self):
        return f"OpenAILanguageModel(model={self.model_name})"

    def __repr__(self):
        return self.__str__()