import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import LanguageModelProvider, LanguageModel
from huggingFaceUtils.models import PATH as MODELS_PATH
from huggingface_hub import login

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)

class HuggingFaceService(LanguageModelProvider):
    def __init__(self, device=None, dtype=None, cache_dir: Path = MODELS_PATH):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.cache_dir = cache_dir
        login(HF_AUTH_TOKEN)

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
    def __init__(self, model_name, system_prompt, device, dtype, cache_dir=None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
            token=HF_AUTH_TOKEN if HF_AUTH_TOKEN else None,
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
    def ask(self, question, max_output=1024, temperature=0.0):
        single = isinstance(question, str)
        questions = [question] if single else question
        prompts = [self._build_prompt(q) for q in questions]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        if temperature == 0.0:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_output,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_output,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True,
        )
        return decoded[0] if single else decoded

    def _build_prompt(self, question: str) -> str:
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        # Fall-back if chat_template is not available:
        try:
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass

        if self.system_prompt:
            return f"{self.system_prompt}\nUser: {question}"
        else:
            return question

    def __str__(self):
        return f"HuggingFaceLanguageModel(model={self.model_name}, device={self.device})"

    def __repr__(self):
        return self.__str__()