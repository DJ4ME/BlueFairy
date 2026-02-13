from pathlib import Path
from bluefairy.norms import run_norms_translation
from bluefairy.nouns import Stakeholder
from core import LanguageModelProvider
from huggingFaceUtils import HuggingFaceService
from ollamaUtils import OLLAMA_URL, OLLAMA_PORT, OllamaService


OLLAMA_MODELS = [
    "qwen2.5:1.5b",
    "stablelm-zephyr:latest",
    "phi3.5:latest",
    "phi3:latest",
    "mistral:latest",
    "openhermes:latest",
    "medllama2:latest",
    "qwen2.5:latest",
    "cyberuser42/DeepSeek-R1-Distill-Llama-8B:latest",
    "llama3:latest"
]

HF_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "stabilityai/stablelm-zephyr-3b",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
]


def get_provider(backend: str) -> LanguageModelProvider:
    backend = backend.lower()

    if backend == "ollama":
        return OllamaService(OLLAMA_URL, OLLAMA_PORT)

    elif backend == "hf":
        return HuggingFaceService()

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def translate_norms(provider: LanguageModelProvider,
                    model_name: str,
                    batch_size: int,
                    norms: list[str],
                    examples: str = "",
                    output_file: Path | None = None
                    ) -> None:
    stakeholder = Stakeholder(model_name)
    stakeholder.norms = norms
    run_norms_translation([stakeholder], provider, batch_size, model_name, examples, output_file)


def sanitize_name(model_name: str) -> str:
    return model_name.replace(':', '_').replace('\\', '_').replace('/', '_')

