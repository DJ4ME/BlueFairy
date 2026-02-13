import torch
import pynvml
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


def compute_safe_batch_size(token_len_per_example: int = 512, safety_factor: float = 0.2) -> int:
    """
    Compute a safe batch size based on actual free GPU memory.
    :param token_len_per_example: average number of tokens per input example
    :param safety_factor: fraction of free memory to actually use (0.5 = 50%)
    :return: batch size (>=1)
    """
    if not torch.cuda.is_available():
        return 1

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_bytes = mem_info.free * safety_factor
    print(f"Free GPU memory: {mem_info.free / (1024**3):.2f} GB, using up to: {free_bytes / (1024**3):.2f} GB for batch processing.")

    # Approximate memory per example (in bytes): float32 * tokens * vocab_embedding_factor
    # Using 4 bytes per float
    # Assume maximum vocab_embedding_factor to be around 4 for safety (depends on model architecture)
    mem_per_example = token_len_per_example * 4 * 4
    batch_size = max(1, int(free_bytes / mem_per_example))
    return batch_size


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
        output_file =PATH / DEFAULT_NOUNS_FILE,
    ) -> None:
    """
    Translates textual norms into logical norms for each stakeholder with optional batching for HF.
    """
    with open(output_file, mode='w', encoding='utf-8') as f:
        f.write(HEADER)

        all_norms = []
        all_stakeholders = []
        for stakeholder in stakeholders:
            for textual_norm in stakeholder.norms:
                all_norms.append(textual_norm)
                all_stakeholders.append(stakeholder.noun)


        batch_size = compute_safe_batch_size()
        print(f"Using batch size of {batch_size} for model: {model_name}")

        for i in range(0, len(all_norms), batch_size):
            batch_norms = all_norms[i:i+batch_size]
            batch_stakeholders = all_stakeholders[i:i+batch_size]

            logical_norms = provider.use(model_name, examples).ask(
                batch_norms,
                max_output=256,
                temperature=DEFAULT_TEMPERATURE,
            )

            if isinstance(logical_norms, str):
                logical_norms = [logical_norms]

            for noun, textual_norm, logical_norm in zip(batch_stakeholders, batch_norms, logical_norms):
                logical_norm = logical_norm.split("\n")[0].strip()
                f.write(f'{noun},"{textual_norm}",{logical_norm}\n')

    print(f"Norms translation completed. Translated norms saved to {output_file}.")