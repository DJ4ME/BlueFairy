from pathlib import Path
import pandas as pd
from evaluation.data import load_test_set
from evaluation.sentenceTranslation.analysis import load_sanitized_result
from evaluation.sentenceTranslation.results import PATH as RESULTS_PATH
from evaluation.unification.utils import initialize_components, create_predicate_merge_mappings

PATH = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # data_set = load_test_set()
    data_set = load_sanitized_result(RESULTS_PATH / "Qwen_Qwen2.5-7B-Instruct_hf_examples.csv")
    data_set = data_set.iloc[:, 1:3]
    data_set.columns = ['NL', 'FOL']
    fol_formulae = data_set['FOL'].tolist()
    alphas = [0.1 * i for i in range(0, 11)]
    thresholds = [0.5 + i * 0.025 for i in range(1, 20)]

    for alpha in alphas:
        predicates, matrix = initialize_components(fol_formulae, alpha=alpha)
        maps = create_predicate_merge_mappings(thresholds, predicates, matrix)
        max_num_of_clusters = 0
        best_threshold = 0.0

        for m in maps:
            data = []
            for (pred_name, arity), (rep_name, _) in m.items():
                if pred_name != rep_name:
                    data.append([int(arity), pred_name, rep_name])
            data = pd.DataFrame(data, columns=['Arity', 'Predicate', 'Representative'])
            threshold_str = str(int(thresholds[maps.index(m)] * 1000))
            data.to_csv(PATH / f'predicate_map_a{str(int(alpha*10))}_t{threshold_str}.csv', index=False)
            # Determine number of clusters with at least one member
            num_of_clusters = len(set(v for (k,_), (v, _) in m.items() if k != v))
            if num_of_clusters > max_num_of_clusters:
                max_num_of_clusters = num_of_clusters
                best_threshold = thresholds[maps.index(m)]
        print(f"Maximum number of clusters ({max_num_of_clusters}) with {alpha:.2f} at threshold {best_threshold:.3f}")