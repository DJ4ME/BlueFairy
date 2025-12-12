from enum import Enum
from pathlib import Path

PATH = Path(__file__).parent.resolve()
PromptTask = Enum("PromptTask", "nouns_collection nouns_resolution norms_identification norms_translation")


def load_system_prompt(prompt_task: PromptTask) -> str:
    prompt_file = PATH / f"{prompt_task.name}.txt"
    with open(prompt_file, "r", encoding="utf-8") as file:
        system_prompt = file.read()
    return system_prompt