import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import LanguageModelProvider, LanguageModel
from huggingFaceUtils.models import PATH as MODELS_PATH
from huggingface_hub import login

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)


def load_system_prompt(prompt_files):
    parts = []
    for file in prompt_files:
        with open(file, encoding="utf-8") as f:
            parts.append(f.read().strip())
    return "\n\n".join(parts)


class HuggingFaceService(LanguageModelProvider):
    def __init__(self, device=None, dtype=None, cache_dir: Path = MODELS_PATH):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.cache_dir = cache_dir
        login(HF_AUTH_TOKEN)

    def use(self, language_model: str, prompt_files=None, system_prompt=None) -> LanguageModel:
        # Compose system prompt from files if no system_prompt provided
        if system_prompt is None and prompt_files is not None:
            if isinstance(prompt_files, (list, tuple)):
                prompt_paths = [str(Path(f)) for f in prompt_files]
            else:
                raise ValueError("prompt_files must be a list of path!")
            system_prompt = load_system_prompt(prompt_paths)
        return HuggingFaceLanguageModel(
            model_name=language_model,
            system_prompt=system_prompt if system_prompt else "",
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

        # Always prefer device_map="auto" for big 70B models
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="auto",  # always auto for big models
        )

        self.model.eval()

    @torch.no_grad()
    def ask(self, question, max_output=4096, temperature=0.1):
        single = isinstance(question, str)
        questions = [question] if single else question
        prompts = [self._build_prompt(q) for q in questions]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192
        ).to(self.device)

        generation_args = dict(
            max_new_tokens=max_output,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0 else 1e-6,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        outputs = self.model.generate(
            **inputs,
            **generation_args
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

        # Prefer chat_template if possible
        try:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass

        # Plain fallback
        if self.system_prompt:
            return f"{self.system_prompt}\n\nUser: {question}"
        else:
            return question

    def __str__(self):
        return f"HuggingFaceLanguageModel(model={self.model_name}, device={self.device})"

    def __repr__(self):
        return self.__str__()