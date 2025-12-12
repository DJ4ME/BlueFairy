from enum import Enum
from pathlib import Path
from bluefairy.nouns.utils import load_norms_from_txt, TextualNorms

PATH = Path(__file__).parent.resolve()
StakeholderName = Enum("StakeholderName", "ministry nutritionist user")

def load_textual_norms(stakeholder: StakeholderName) -> TextualNorms:
    file_path = PATH / f"{stakeholder.name.lower()}-textual-norms.txt"
    return load_norms_from_txt(str(file_path))
