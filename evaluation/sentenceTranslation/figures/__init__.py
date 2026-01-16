from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt
from evaluation.analysis import PATH as ANALYSIS_PATH


PATH = Path(__file__).parent.resolve()
MATCH_EXPECTED = "match_expected"
SYNTAX_VALID = "syntax_valid"
MODELS_PRETTY_NAMES = {
    "qwen2.5_latest": "Qwen 2.5 (7.6B)",
    "stablelm-zephyr_latest": "StableLM Zephyr (3B)",
    "phi3.5_latest": "Phi 3.5 (3.8B)",
    "phi3_latest": "Phi 3 (4B)",
    "mistral_latest": "Mistral (7B)",
    "openhermes_latest": "OpenHermes (7B)",
    "medllama2_latest": "MedLlama 2 (7B)",
    "qwen2.5_1.5b": "Qwen 2.5 (1.5B)",
    "cyberuser42_DeepSeek-R1-Distill-Llama-8B_latest": "DeepSeek R1 Distill Llama (8B)",
    "llama3_latest": "Llama 3 (8B)"
}


def plot_llm_performance(
        data: pd.DataFrame,
        y_axis_title: str,
        output_file: str or Path
) -> None:
    """
    Generate a pdf plot showing LLM performance.
    Each column in the DataFrame represents an LLM model,
    and each row represents if the model's output was correct (true) or incorrect (false).
    Results of the same model but different configurations (_examples and _no_examples) are grouped together.
    Y-axis: Percentage of correct outputs
    X-axis: LLM models by name
    :param data: DataFrame with LLM performance data
    :param y_axis_title: the title for the y axis
    :param output_file: the output file path for the pdf plot
    :return: none
    """
    performance = {}
    for column in data.columns:
        correct_count = data[column].sum()
        total_count = len(data[column])
        performance[column] = (correct_count / total_count) * 100  # percentage

    plt.clf()
    plt.figure(figsize=(12, 7))

    # Group results by model and configuration
    grouped_performance = {}
    for key, value in performance.items():
        base_key = key.replace("validation_", "").replace("_no_examples", "").replace("_examples", "")
        if base_key not in grouped_performance:
            grouped_performance[base_key] = {}
        if "_no_examples" in key:
            grouped_performance[base_key]["no_examples"] = value
        elif "_examples" in key:
            grouped_performance[base_key]["examples"] = value

    # Prepare data for plotting
    models = []
    examples_values = []
    no_examples_values = []
    for model, configs in grouped_performance.items():
        models.append(MODELS_PRETTY_NAMES.get(model, model))
        examples_values.append(configs.get("examples", 0))
        no_examples_values.append(configs.get("no_examples", 0))

    # Plot bars
    x = range(len(models))
    bar_width = 0.4
    plt.bar([i - bar_width / 2 for i in x], examples_values, width=bar_width, color='blue', label='Examples')
    plt.bar([i + bar_width / 2 for i in x], no_examples_values, width=bar_width, color='orange', label='No Examples')

    # Add labels and legend
    plt.xlabel('LLM Models')
    plt.ylabel(y_axis_title)
    plt.title('LLM Performance Comparison')
    plt.ylim(0, min(100, max(examples_values + no_examples_values) + 10))
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def merge_results(files: Iterable, metric: str) -> pd.DataFrame:
    results = pd.DataFrame()
    for result_file in files:
        model_name = result_file.stem
        df = pd.read_csv(result_file)
        if metric in df.columns:
            df_model = df[[metric]].rename(columns={metric: model_name})
            if results.empty:
                results = df_model
            else:
                results = results.join(df_model, how='outer')
    return results


if __name__ == "__main__":
    # Read all csv files in analysis/results
    all_result_files = list(ANALYSIS_PATH.glob("*.csv"))
    # Merge all results into a single DataFrame using index
    # Keep the shared column MATCH_EXPECTED and rename using the model name
    merged_results_match = merge_results(all_result_files, MATCH_EXPECTED)
    merged_results_syntax = merge_results(all_result_files, SYNTAX_VALID)
    plot_llm_performance(merged_results_match, "Percentage of correct matching formulas", PATH / "llm_performance.pdf")
    plot_llm_performance(merged_results_syntax, "Percentage of syntactically valid formulas", PATH / "llm_syntax_performance.pdf")
