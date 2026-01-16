import re
import pandas as pd
from io import StringIO
from pathlib import Path

from evaluation.analysis.malls_fol_parser import is_valid_fol_malls
from evaluation.data import load_test_set
from evaluation.results import PATH as RESULTS_PATH
from evaluation.analysis.utils import compare_fol, parse_fol, is_valid_fol, is_valid_fol_closed

PATH = Path(__file__).parent.resolve()

def load_sanitized_result(file_path: Path) -> pd.DataFrame:
    """
    Sanitizes the results file to ensure that the LogicalNorm column is properly quoted.
    The original file is not modified; instead, a sanitized DataFrame is returned.
    :param file_path: the path to the results file
    :return: the sanitized DataFrame
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[0].strip()
    sanitized_lines = [header + '\n']
    for line in lines[1:]:
        pattern = r'^(.*?),(".*?"),(\".*\"|[^"].*[^"]|".*|.*)$'
        match = re.match(pattern, line.strip())
        if match:
            stakeholder = match.group(1).strip()
            textual_norm = match.group(2).strip()
            logical_norm = match.group(3).strip()
            # Ensure logical_norm is properly quoted
            if logical_norm.startswith("'"):
                logical_norm = logical_norm.replace("'", '"')
            if logical_norm.endswith("'"):
                logical_norm = logical_norm.replace("'", '"')
            if logical_norm.endswith(',"'):
                logical_norm += '"'
            if logical_norm.startswith('"') and not logical_norm.endswith('"'):
                logical_norm = f'{logical_norm}"'
            elif logical_norm.endswith('"') and not logical_norm.startswith('"'):
                logical_norm = f'"{logical_norm}'
            elif not logical_norm.startswith('"') and not logical_norm.endswith('"'):
                logical_norm = f'"{logical_norm}"'
            if logical_norm.count('"') % 2 != 0:
                logical_norm = logical_norm[:-1]
            sanitized_line = f'{stakeholder},{textual_norm},{logical_norm}\n'
            sanitized_lines.append(sanitized_line)
        else:
            sanitized_lines.append(line)
    sanitized_content = ''.join(sanitized_lines)
    sanitized_df = pd.read_csv(StringIO(sanitized_content))
    return sanitized_df


def lowercase_constants(formula: str) -> str:
    pattern = r'\b([A-Z][A-Za-z0-9_]*)\b(?!\()'

    def repl(match):
        word = match.group(1)
        return word.lower()

    return re.sub(pattern, repl, formula)


def validate_model_translations(prediction: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the model translations by comparing predictions with expected results.
    Also checks for syntax correctness in the logical norms.
    :param prediction: the DataFrame containing model predictions
    :param expected: the DataFrame containing expected results
    :return: a DataFrame with validation results (two extra columns: 'match_expected', 'syntax_valid')
    """
    # Merge by index the two dataframes into one for comparison
    # Prediction header: Stakeholder,TextualNorm,LogicalNorm
    # Expected header: NL,FOL
    merged = prediction.merge(expected, left_index=True, right_index=True, suffixes=('_pred', '_exp'))
    # Keep only TextualNorm (equivalent to NL), LogicalNorm must become PredictedFOL, FOL must become ExpectedFOL
    merged = merged[['TextualNorm', 'LogicalNorm', 'FOL']]
    merged = merged.rename(columns={
        'TextualNorm': 'TextualNorm',
        'LogicalNorm': 'PredictedFOL',
        'FOL': 'ExpectedFOL'
    })
    # Compare PredictedFOL with ExpectedFOL
    merged['match_expected'] = merged.apply(
        lambda row: compare_fol(row['PredictedFOL'], row['ExpectedFOL']),
        axis=1
    )
    # Check syntax correctness of PredictedFOL
    merged['syntax_valid'] = merged['PredictedFOL'].apply(
        lambda fol: is_valid_fol_closed(fol)
    )
    # Check syntax correctness using the original MALLS dataset parser
    merged['syntax_valid_malls'] = merged['PredictedFOL'].apply(
        lambda fol: is_valid_fol_malls(fol)
    )
    return merged


if __name__ == "__main__":
    all_result_files = RESULTS_PATH.glob("*.csv")
    expected_df = load_test_set()
    # Apply lowercase_constants to all expected FOL formulas in the test set
    expected_df['FOL'] = expected_df['FOL'].apply(lowercase_constants)
    for result_file in all_result_files:
        print(f"Validating results in file: {result_file.name}")
        predictions_df = load_sanitized_result(result_file)
        validation_df = validate_model_translations(predictions_df, expected_df)
        # Print accuracy summary
        total = len(validation_df)
        correct = validation_df['match_expected'].sum()
        syntax_correct = validation_df['syntax_valid'].sum()
        print(f"Total samples: {total}")
        print(f"Correctly matched expected: {correct} ({(correct / total) * 100:.2f}%)")
        print(f"Syntax valid: {syntax_correct} ({(syntax_correct / total) * 100:.2f}%)")
        print(f"Syntax valid (MALLS parser): {validation_df['syntax_valid_malls'].sum()} ({(validation_df['syntax_valid_malls'].sum() / total) * 100:.2f}%)")
        # Save validation results
        validation_file = PATH / f"validation_{result_file.name}"
        validation_df.to_csv(validation_file, index=False)
        print(f"Validation completed. Results saved to: {validation_file}\n")