from bluefairy.norms import run_norms_translation
from bluefairy.nouns import Stakeholder
from core import LanguageModelProvider
from evaluation.data import load_examples, load_test_set
from evaluation.results import PATH as RESULTS_PATH
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


def translate_norms(provider: LanguageModelProvider,
                    model_name: str,
                    norms: list[str],
                    examples: str = "",
                    output_file = None
                    ) -> None:
    stakeholder = Stakeholder(model_name)
    stakeholder.norms = norms
    run_norms_translation([stakeholder], provider, model_name, examples, output_file)


if __name__ == "__main__":
    test_set = load_test_set()
    examples_df = load_examples()
    examples_txt = examples_df.apply(
        lambda row: f"Textual Norm: {row['NL']}\nLogical Norm: {row['FOL']}\n", axis=1
    ).str.cat(sep="\n")

    # With examples
    for llm in LLM_FOR_TESTING:
        print(f"Running norms translation tests with examples for model: {llm}")
        translate_norms(
            provider=OllamaService(OLLAMA_URL, OLLAMA_PORT),
            model_name=llm,
            norms=test_set['NL'].tolist(),
            examples=examples_txt,
            output_file=RESULTS_PATH / f"{llm.replace(':', '_')}_examples.csv"
        )
        print(f"Completed norms translation tests for model: {llm}\n\n")

    # Without examples
    for llm in LLM_FOR_TESTING:
        print(f"Running norms translation tests without examples for model: {llm}")
        translate_norms(
            provider=OllamaService(OLLAMA_URL, OLLAMA_PORT),
            model_name=llm,
            norms=test_set['NL'].tolist(),
            examples="",
            output_file=RESULTS_PATH / f"{llm.replace(':', '_')}_no_examples.csv"
        )
        print(f"Completed norms translation tests for model: {llm}\n\n")
