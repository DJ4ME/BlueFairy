import argparse
from pathlib import Path
from bluefairy.norms import run_norms_translation
from bluefairy.nouns import Stakeholder
from core import LanguageModelProvider
from evaluation.data import load_examples, load_test_set
from evaluation.sentenceTranslation.results import PATH as RESULTS_PATH
from huggingFaceUtils import HuggingFaceService
from ollamaUtils import OLLAMA_URL, OLLAMA_PORT, OllamaService


LLM_FOR_TESTING = [
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
                    norms: list[str],
                    examples: str = "",
                    output_file: Path | None = None
                    ) -> None:
    stakeholder = Stakeholder(model_name)
    stakeholder.norms = norms
    run_norms_translation([stakeholder], provider, model_name, examples, output_file)


def sanitize_name(model_name: str) -> str:
    return model_name.replace(':', '_').replace('\\', '_').replace('/', '_')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["ollama", "hf"],
        help="Backend to use: ollama or hf"
    )

    args = parser.parse_args()

    provider = get_provider(args.backend)

    test_set = load_test_set()
    examples_df = load_examples()

    examples_txt = examples_df.apply(
        lambda row: f"Textual Norm: {row['NL']}\nLogical Norm: {row['FOL']}\n",
        axis=1
    ).str.cat(sep="\n")

    print(f"\nUsing backend: {args.backend}")
    print(f"Provider: {provider}\n")

    # ======================
    # WITH EXAMPLES
    # ======================

    for llm in LLM_FOR_TESTING:

        pattern_name = sanitize_name(llm)
        file = RESULTS_PATH / f"{pattern_name}_{args.backend}_examples.csv"

        if file.exists():
            print(f"Skipping (examples) for model: {llm} — file exists.")
            continue

        print(f"Running (examples) for model: {llm}")

        translate_norms(
            provider=provider,
            model_name=llm,
            norms=test_set['NL'].tolist(),
            examples=examples_txt,
            output_file=file
        )

        print(f"Completed (examples) for model: {llm}\n")

    # ======================
    # WITHOUT EXAMPLES
    # ======================

    for llm in LLM_FOR_TESTING:

        pattern_name = sanitize_name(llm)
        file = RESULTS_PATH / f"{pattern_name}_{args.backend}_no_examples.csv"

        if file.exists():
            print(f"Skipping (no examples) for model: {llm} — file exists.")
            continue

        print(f"Running (no examples) for model: {llm}")

        translate_norms(
            provider=provider,
            model_name=llm,
            norms=test_set['NL'].tolist(),
            examples="",
            output_file=file
        )

        print(f"Completed (no examples) for model: {llm}\n")
