from pathlib import Path
import pandas as pd


PATH = Path(__file__).parent.resolve()


def load_examples() -> pd.DataFrame:
    file_path = PATH / "MALLS-nutrition-examples.csv"
    return pd.read_csv(file_path)

def load_test_set() -> pd.DataFrame:
    file_path = PATH / "MALLS-nutrition-test.csv"
    return pd.read_csv(file_path)