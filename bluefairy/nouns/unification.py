import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from bluefairy.grammar.utils import parse_or_false, PRED_KEY
from bluefairy.nouns.embedding import build_embedding_space, build_predicate_similarity_matrix, \
    generate_predicate_embedding_sentences, generate_constant_embedding_sentences, build_constant_similarity_matrix
from bluefairy.nouns.graph import build_predicate_unification_map, build_constant_unification_map
from bluefairy.nouns.lexical_metrics import build_predicate_lexical_similarity_matrix, \
    build_constant_lexical_similarity_matrix


def collect_predicates(list_of_fol_formulas: list[str]) -> dict[PRED_KEY, int]:
    """
    Collects unique predicates name and their arity from a list of first-order logic formulas.
    :param list_of_fol_formulas: the list of first-order logic formulas
    :return: a dictionary of unique predicates and their occurrence count
    """
    predicates: dict[PRED_KEY, int] = dict()
    not_parsed = 0
    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            not_parsed += 1
            print(fol)
        else:
            for pred_name, arity in parsed.get_predicates():
                key = (pred_name, arity)
                if key not in predicates:
                    predicates[key] = 0
                predicates[key] += 1
    if not_parsed > 0:
        print(f"Warning: {not_parsed} formulas could not be parsed.")
    return predicates


def collect_constants(list_of_fol_formulas: list[str]) -> dict[str, int]:
    """
    Collects unique constant names from a list of first-order logic formulas.
    :param list_of_fol_formulas: the list of first-order logic formulas
    :return: a dictionary of unique constant names and their occurrence count
    """
    elements: dict[str, int] = dict()
    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            continue
        for element in parsed.get_constants():
            if element not in elements:
                elements[element] = 0
            elements[element] += 1
    return elements


def create_predicate_arity_matrix(predicates: list[PRED_KEY]) -> pd.DataFrame:
    """
    Creates a predicate arity matrix predicate x predicate from the list of predicates and their arities.
    A cell is 1 if the predicates have the same arity, 0 otherwise.
    :param predicates: the list of predicates with their arities
    :return: a DataFrame representing the predicate arity matrix
    """
    data = {pred_key: [] for pred_key in predicates}
    index = []
    for pred_key_1 in predicates:
        index.append(pred_key_1)
        arity_1 = pred_key_1[1]
        for pred_key_2 in predicates:
            arity_2 = pred_key_2[1]
            data[pred_key_2].append(1 if arity_1 == arity_2 else 0)
    df = pd.DataFrame(data, index=index)
    df.columns = df.index
    return df


def create_predicate_terms_matrices(list_of_fol_formulas: list[str]) -> list[pd.DataFrame]:
    """
    Creates a list of predicate terms matrices predicates x terms from a list of first-order logic formulas.
    Each matrix corresponds to one specific position of the predicate terms.
    :param list_of_fol_formulas: the list of first-order logic formulas
    :return: a list of DataFrames representing predicate terms matrices.
    1 if term is in predicate terms corresponding position, 0 otherwise.
    """
    def is_variable(term: str) -> bool:
        return term and len(term) == 1

    pred_pos_terms: dict[PRED_KEY, dict[int, set]] = defaultdict(lambda: defaultdict(set))

    predicates = set()
    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            continue
        for (pred_name, arity), terms in parsed.get_full_predicates():
            pred_key: PRED_KEY = (pred_name, arity)
            predicates.add(pred_key)
            for pos, term in enumerate(terms):
                if is_variable(term):
                    continue
                pred_pos_terms[pred_key][pos].add(term)

    if not pred_pos_terms:
        return []

    predicates = sorted(list(predicates))
    all_terms = sorted({t for pos_dict in pred_pos_terms.values() for ts in pos_dict.values() for t in ts})
    max_arity = max(arity for _, arity in predicates)

    df_list = []
    for pos in range(max_arity):
        data = {term: [] for term in all_terms}
        index = []
        for pred_key in predicates:
            index.append(pred_key)
            row_terms = pred_pos_terms[pred_key].get(pos, set())
            for term in all_terms:
                data[term].append(1 if term in row_terms else 0)
        df = pd.DataFrame(data, index=index)
        df_list.append(df)

    return df_list


def compute_similarity_scores(semantic_scores: pd.DataFrame, lexical_scores: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Computes combined similarity scores from semantic and lexical similarity scores.
    :param semantic_scores: the semantic similarity scores DataFrame
    :param lexical_scores: the lexical similarity scores DataFrame
    :param alpha: the weight for lexical similarity
    :return: a DataFrame with combined similarity scores
    """
    combined_scores = (1 - alpha) * semantic_scores + alpha * lexical_scores
    return combined_scores


def rewrite_formulae(
        list_of_fol_formulas: list[str],
        predicate_unification_map: dict[PRED_KEY, PRED_KEY],
        constant_unification_map: dict[str, str]
) -> list[str]:
    """
    Rewrites a list of first-order logic formulas based on a unification map of predicates.
    :param list_of_fol_formulas: the list of first-order logic formulas
    :param predicate_unification_map: the unification map of predicates
    :param constant_unification_map: the unification map of constants
    """
    rewritten_formulas = []
    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            rewritten_formulas.append(fol)
            continue
        for (pred_name, arity), terms in parsed.get_full_predicates():
            key = (pred_name, arity)
            if key in predicate_unification_map:
                new_pred_name = predicate_unification_map[key][0]
                parsed.rename_predicate(pred_name, new_pred_name, arity)
        for const in parsed.get_constants():
            if const in constant_unification_map:
                new_const = constant_unification_map[const]
                parsed.rename_constant(const, new_const)
        rewritten_formulas.append(parsed.to_readable_string())
    return rewritten_formulas


def generate_predicate_similarity_matrix(
        predicates: dict[PRED_KEY, int],
        matrices: list[pd.DataFrame],
        model: SentenceTransformer,
        predicate_alpha: float = 0.1,
) -> pd.DataFrame:
    """
    Generates a predicate similarity matrix from a list of first-order logic formulas.
    :param predicates: the dictionary of unique predicates and their occurrence count
    :param matrices: the list of predicate terms matrices
    :param model: the sentence transformer model to use for embeddings
    :param predicate_alpha: the weight for lexical similarity
    :return: a DataFrame representing the predicate similarity matrix
    """
    predicate_sentences = generate_predicate_embedding_sentences(matrices)

    arity_predicate_matrix = create_predicate_arity_matrix(list(predicates.keys()))

    embedding_predicate_space = build_embedding_space(list(predicate_sentences.values()),
                                                      lambda x: model.encode(x, normalize_embeddings=False))
    semantic_predicate_matrix_score = build_predicate_similarity_matrix(embedding_predicate_space, predicate_sentences)

    lexical_predicate_matrix_score = build_predicate_lexical_similarity_matrix(list(predicates.keys()))

    predicate_similarity_score = arity_predicate_matrix * compute_similarity_scores(semantic_predicate_matrix_score,
                                                                                    lexical_predicate_matrix_score,
                                                                                    alpha=predicate_alpha)
    return predicate_similarity_score


def generate_constant_similarity_matrix(
        constants: dict[str, int],
        matrices: list[pd.DataFrame],
        model: SentenceTransformer,
        constant_alpha: float = 0.1,
) -> pd.DataFrame:
    """
    Generates a constant similarity matrix from a list of first-order logic formulas.
    :param constants: the dictionary of unique constants and their occurrence count
    :param matrices: the list of predicate terms matrices
    :param model: the sentence transformer model to use for embeddings
    :param constant_alpha: the weight for lexical similarity
    :return: a DataFrame representing the constant similarity matrix
    """
    constant_sentences = generate_constant_embedding_sentences(matrices)

    embedding_constant_space = build_embedding_space(list(constant_sentences.values()),
                                                     lambda x: model.encode(x, normalize_embeddings=False))
    semantic_constant_matrix_score = build_constant_similarity_matrix(embedding_constant_space, constant_sentences)

    lexical_constant_matrix_score = build_constant_lexical_similarity_matrix(
        [const for const, occurrence in constants.items()])

    constant_similarity_score = compute_similarity_scores(semantic_constant_matrix_score, lexical_constant_matrix_score,
                                                          alpha=constant_alpha)
    return constant_similarity_score


def uniformize_formulae(
        list_of_fol_formulae: list[str],
        predicate_alpha: float = 0.1,
        constant_alpha: float = 0.1,
        predicate_threshold: float = 0.8,
        constant_threshold: float = 0.8,
        stats: bool = False
) -> list[str]:
    """
    Uniformizes a list of first-order logic formulas by unifying similar predicates and constants.
    :param list_of_fol_formulae: the list of first-order logic formulas
    :param predicate_alpha: the weight for lexical similarity in predicate unification
    :param constant_alpha: the weight for lexical similarity in constant unification
    :param predicate_threshold: the threshold for predicate unification
    :param constant_threshold: the threshold for constant unification
    :param stats: whether to collect and print statistics
    :return: a list of uniformized first-order logic formulas
    """
    predicates = collect_predicates(list_of_fol_formulae)
    constants = collect_constants(list_of_fol_formulae)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    matrices = create_predicate_terms_matrices(list_of_fol_formulae)
    predicate_similarity_score = generate_predicate_similarity_matrix(predicates, matrices, model, predicate_alpha)
    constant_similarity_score = generate_constant_similarity_matrix(constants, matrices, model, constant_alpha)

    unified_predicates = build_predicate_unification_map(predicate_similarity_score, predicates, threshold=predicate_threshold)
    unified_constants = build_constant_unification_map(constant_similarity_score, constants, threshold=constant_threshold)

    if stats:
        print(f"Number of predicate unifications: {sum(k != v for k, v in unified_predicates.items())} / {len(unified_predicates)}")
        print(f"Number of constant unifications: {sum(k != v for k, v in unified_constants.items())} / {len(unified_constants)}")

    return rewrite_formulae(list_of_fol_formulae, unified_predicates, unified_constants)



if __name__ == "__main__":
    test_formulas = [
        '∀x (Person(x) → Eats(x, apple))',
        '∀x ∀y (Person(x) ∧ Food(y) → Likes(x, y))',
        '∃x (Person(x) ∧ HasDiet(x, vegetarian))',
        '∀x ∀y (Food(y) ∧ Ingredient(y, sugar) → Contains(y, sugar))',
        '∀x ∀y (Person(x) ∧ Food(y) → AllergicTo(x, y) → ¬Eats(x, y))',
        '∃x (Recipe(x) ∧ Uses(x, flour) ∧ Uses(x, sugar))',
        '∀x (Person(x) → Needs(x, protein))',
        '∀x ∀y (Person(x) ∧ Meal(y) → Eats(x, y) → Contains(y, protein))',
        '∃x (Person(x) ∧ FavoriteFood(x, pizza))',
        '∀x ∀y (Person(x) ∧ Food(y) → Prefers(x, y) ↔ Likes(x, y))'
    ]

    rewritten_formulas = uniformize_formulae(test_formulas)
    for original, rewritten in zip(test_formulas, rewritten_formulas):
        print(f"O: {original}")
        print(f"R: {rewritten}\n")



