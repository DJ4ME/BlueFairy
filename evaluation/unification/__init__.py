from pathlib import Path
import pandas as pd
from evaluation.data import load_test_set
from bluefairy.nouns.unification import uniformize_formulae

PATH = Path(__file__).parent.resolve()


if __name__ == "__main__":

    test_set = load_test_set()
    fol_formulae = test_set['FOL'].tolist()
    unified_formulae = uniformize_formulae(
        fol_formulae,
        predicate_threshold=0.9,
        predicate_alpha=0.0,
        stats=True
    )

    df = pd.DataFrame({
        'Original FOL': fol_formulae,
        'Unified FOL': unified_formulae
    })

    df.to_csv(PATH / 'unified_fol_formulae.csv', index=False)

