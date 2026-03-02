from pathlib import Path
from bluefairy.norms import run_norms_translation
from bluefairy.nouns import Stakeholder
from core import LanguageModelProvider
from huggingFaceUtils import HuggingFaceService
from ollamaUtils import OLLAMA_URL, OLLAMA_PORT, OllamaService
from openAIUtils import OpenAIService


def get_provider(backend: str) -> LanguageModelProvider:
    backend = backend.lower()

    if backend == "ollama":
        return OllamaService(OLLAMA_URL, OLLAMA_PORT)

    elif backend == "hf":
        return HuggingFaceService()

    elif backend == "OpenAI":
        return OpenAIService()

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

