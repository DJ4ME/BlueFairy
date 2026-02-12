import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import LanguageModelProvider, LanguageModel
from huggingFaceUtils.models import PATH as MODELS_PATH


class HuggingFaceService(LanguageModelProvider):
    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: Path = MODELS_PATH,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.cache_dir = cache_dir

    def use(self, language_model: str, system_prompt: str) -> LanguageModel:
        return HuggingFaceLanguageModel(
            model_name=language_model,
            system_prompt=system_prompt,
            device=self.device,
            dtype=self.dtype,
            cache_dir=str(self.cache_dir),
        )

    def __str__(self):
        return f"HuggingFaceService(device={self.device}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        device: str,
        dtype: torch.dtype,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        self.model.eval()

    @torch.no_grad()
    def ask(
        self,
        question: str | list[str],
        max_output: int = 1024,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ) -> str | list[str]:

        if isinstance(question, str):
            questions = [question]
            single = True
        else:
            questions = question
            single = False

        prompts = [
            self._build_prompt(q)
            for q in questions
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_output,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return decoded[0] if single else decoded

    def _build_prompt(self, question: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{question}"
        return question

    def __str__(self):
        return f"HuggingFaceLanguageModel(model={self.model_name}, device={self.device})"

    def __repr__(self):
        return self.__str__()
