from pathlib import Path
import pandas as pd
from bluefairy.nouns.unification import uniformize_formulae
from evaluation.sentenceTranslation.results import PATH as RESULTS_PATH
from evaluation.sentenceTranslation.analysis import load_sanitized_result
from bluefairy.grammar.malls_fol_parser import is_valid_fol_malls

PATH = Path(__file__).parent.resolve()


if __name__ == "__main__":

    test_set = load_sanitized_result(RESULTS_PATH / "Qwen_Qwen2.5-7B-Instruct_hf_examples.csv")
    test_set = test_set.iloc[:, 1:3]
    test_set.columns = ['NL', 'FOL']
    fol_formulae = test_set['FOL'].tolist()
    # Check syntax correctness using the original MALLS dataset parser
    parsable_fol_formulae = test_set['FOL'].apply(
        lambda fol: is_valid_fol_malls(fol)
    )
    fol_formulae = [fol for fol, valid in zip(fol_formulae, parsable_fol_formulae) if valid]
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

