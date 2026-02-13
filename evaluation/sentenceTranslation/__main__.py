from evaluation.data import load_examples, load_test_set
import argparse
from evaluation.sentenceTranslation.results import PATH as RESULTS_PATH
from evaluation.sentenceTranslation import get_provider, sanitize_name, translate_norms, HF_MODELS, OLLAMA_MODELS

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["ollama", "hf"],
        help="Backend to use: ollama or hf"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for HuggingFace models (ignored for Ollama)"
    )

    args = parser.parse_args()

    provider = get_provider(args.backend)
    batch_size = args.batch_size if args.backend == "hf" else 1

    test_set = load_test_set()
    examples_df = load_examples()

    examples_txt = examples_df.apply(
        lambda row: f"Textual Norm: {row['NL']}\nLogical Norm: {row['FOL']}\n",
        axis=1
    ).str.cat(sep="\n")

    print(f"\nUsing backend: {args.backend}")
    print(f"Provider: {provider}\n")

    # ======================
    # WITH EXAMPLES
    # ======================

    if args.backend == "hf":
        models_to_test = HF_MODELS
    else:
        models_to_test = OLLAMA_MODELS

    for llm in models_to_test:

        pattern_name = sanitize_name(llm)
        file = RESULTS_PATH / f"{pattern_name}_{args.backend}_examples.csv"

        if file.exists():
            print(f"Skipping (examples) for model: {llm} — file exists.")
            continue

        print(f"Running (examples) for model: {llm}")

        translate_norms(
            provider=provider,
            model_name=llm,
            batch_size=batch_size,
            norms=test_set['NL'].tolist(),
            examples=examples_txt,
            output_file=file
        )

        print(f"Completed (examples) for model: {llm}\n")

    # ======================
    # WITHOUT EXAMPLES
    # ======================

    for llm in models_to_test:

        pattern_name = sanitize_name(llm)
        file = RESULTS_PATH / f"{pattern_name}_{args.backend}_no_examples.csv"

        if file.exists():
            print(f"Skipping (no examples) for model: {llm} — file exists.")
            continue

        print(f"Running (no examples) for model: {llm}")

        translate_norms(
            provider=provider,
            model_name=llm,
            batch_size=batch_size,
            norms=test_set['NL'].tolist(),
            examples="",
            output_file=file
        )

        print(f"Completed (no examples) for model: {llm}\n")