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
    """
    Translates a textual norm into a logical norm using an LLM.
    :param provider: the language model provider to use
    :param model_name: the name of the language model to use
    :param norm: the textual norm to translate
    :param examples: examples of translations
    :return: the logical norm
    """

    system_prompt = load_system_prompt(PromptTask.norms_translation)
    system_prompt = system_prompt.replace(
        "{FEW_SHOT_EXAMPLES}" if examples != "" else "\nEXAMPLES\n{FEW_SHOT_EXAMPLES}\n---",
        examples
    )
    model = provider.use(OLLAMA_MODEL if model_name == "" else model_name, system_prompt)
    print(f"Translating textual norm to logical norm: {norm}")
    response = model.ask(norm, max_output=256, temperature=DEFAULT_TEMPERATURE)
    # ignore everything after the first newline
    response = response.split('\n')[0]
    print(f"Translation completed: {response}")
    return response.strip()


def run_norms_translation(
        stakeholders: list[Stakeholder],
        provider: LanguageModelProvider,
        model_name: str = "",
        examples: str = "",
        output_file =PATH / DEFAULT_NOUNS_FILE) -> None:
    """
    Step 2, The function translates textual norms into logical norms for each stakeholder.
    :param stakeholders: the list of stakeholders with their textual norms
    :param provider: the language model provider to use
    :param model_name: the name of the language model to use
    :param examples: examples of translations
    :param output_file: the norms file path
    :return: none
    """

    with open(output_file, mode='w', encoding='utf-8') as f:
        f.write(HEADER)
        for stakeholder in stakeholders:
            for textual_norm in stakeholder.norms:
                logical_norm = textual_norm_to_logic_norm(provider, model_name, textual_norm, examples)
                f.write(f'{stakeholder.noun},"{textual_norm}",{logical_norm}\n')
    print(f"Norms translation completed. Translated norms saved to {output_file}.")
