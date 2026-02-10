from sentence_transformers import SentenceTransformer

from bluefairy.grammar.utils import PRED_KEY
from bluefairy.nouns.graph import build_predicate_unification_map
from bluefairy.nouns.unification import collect_predicates, create_predicate_terms_matrices, \
    generate_predicate_similarity_matrix


MODEL_NAME = 'all-MiniLM-L6-v2'
TRANSFORMER = SentenceTransformer(MODEL_NAME)


def initialize_components(fol_formulae: list[str], alpha: float):
    print(f'Initializing components from {len(fol_formulae)} FOL formulae')
    predicates = collect_predicates(fol_formulae)
    matrices = create_predicate_terms_matrices(fol_formulae)
    matrix = generate_predicate_similarity_matrix(predicates, matrices, TRANSFORMER, alpha)
    return predicates, matrix


def create_predicate_merge_mappings(thresholds: list[float], predicates, matrix) -> list[dict[PRED_KEY, PRED_KEY]]:
    merge_mappings = []
    for threshold in thresholds:
        merge_map = build_predicate_unification_map(
            sim_matrix=matrix,
            occurrences=predicates,
            threshold=threshold
        )
        merge_mappings.append(merge_map)
    return merge_mappings
