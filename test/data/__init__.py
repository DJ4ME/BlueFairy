from enum import Enum
from pathlib import Path
import pandas as pd
from bluefairy.nouns.utils import load_norms_from_txt, TextualNorms


PATH = Path(__file__).parent.resolve()
StakeholderName = Enum("StakeholderName", "ministry nutritionist user")


def load_textual_norms(stakeholder: StakeholderName) -> TextualNorms:
    file_path = PATH / f"{stakeholder.name.lower()}-textual-norms.txt"
    return load_norms_from_txt(str(file_path))

def load_examples() -> pd.DataFrame:
    file_path = PATH / "MALLS-nutrition-examples.csv"
    return pd.read_csv(file_path)
