from bluefairy.nouns import Stakeholder
from bluefairy.nouns.utils import TextualNorm, LogicalNorm
from bluefairy.prompts import load_system_prompt, PromptTask, PATH
from core import LanguageModelProvider
from ollamaUtils import OllamaService, OLLAMA_URL, OLLAMA_PORT


OLLAMA_SERVICE = OllamaService(OLLAMA_URL, OLLAMA_PORT)
OLLAMA_MODEL = "phi3.5:latest"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NOUNS_FILE = "norms_translation.csv"
HEADER = "Stakeholder,TextualNorm,LogicalNorm\n"


def textual_norm_to_logic_norm(
        provider: LanguageModelProvider,
        model_name: str = "",
        norm: TextualNorm = "",
        examples: str = ""
    ) -> LogicalNorm:

    system_prompt = load_system_prompt(PromptTask.norms_translation)
    system_prompt = system_prompt.replace(
        "{FEW_SHOT_EXAMPLES}" if examples != "" else "\nEXAMPLES\n{FEW_SHOT_EXAMPLES}\n---",
        examples
    )

    model = provider.use(
        OLLAMA_MODEL if model_name == "" else model_name,
        system_prompt
    )

    response = model.ask(norm, max_output=256, temperature=DEFAULT_TEMPERATURE)
    response = response.split('\n')[0]
    return response.strip()


def run_norms_translation(
        stakeholders: list[Stakeholder],
        provider: LanguageModelProvider,
        model_name: str = "",
        examples: str = "",
        output_file=PATH / DEFAULT_NOUNS_FILE
    ) -> None:

    system_prompt = load_system_prompt(PromptTask.norms_translation)
    system_prompt = system_prompt.replace(
        "{FEW_SHOT_EXAMPLES}" if examples != "" else "\nEXAMPLES\n{FEW_SHOT_EXAMPLES}\n---",
        examples
    )

    model = provider.use(
        OLLAMA_MODEL if model_name == "" else model_name,
        system_prompt
    )

    with open(output_file, mode='w', encoding='utf-8') as f:
        f.write(HEADER)

        for stakeholder in stakeholders:
            norms = stakeholder.norms

            if not norms:
                continue

            responses = model.ask(
                norms,
                max_output=256,
                temperature=DEFAULT_TEMPERATURE
            )

            for textual_norm, logical_norm in zip(norms, responses):
                logical_norm = logical_norm.split('\n')[0].strip()
                f.write(f'{stakeholder.noun},"{textual_norm}",{logical_norm}\n')

    print(f"Norms translation completed. Translated norms saved to {output_file}.")