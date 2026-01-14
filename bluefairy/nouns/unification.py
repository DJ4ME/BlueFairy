import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from bluefairy.grammar.utils import parse_or_false, PRED_KEY, predicate_similarity_score
from bluefairy.nouns.embedding import build_embedding_space, build_predicate_similarity_matrix, \
    generate_predicate_embedding_sentences, generate_constant_embedding_sentences
from bluefairy.nouns.graph import build_unification_map
from bluefairy.nouns.lexical_metrics import build_predicate_lexical_similarity_matrix, \
    build_constant_lexical_similarity_matrix


def collect_predicates(list_of_fol_formulas: list[str]) -> dict[PRED_KEY, int]:
    """
    Collects unique predicates name and their arity from a list of first-order logic formulas.
    :param list_of_fol_formulas: the list of first-order logic formulas
    :return: a dictionary of unique predicates and their occurrence count
    """
    predicates: dict[PRED_KEY, int] = dict()
    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            continue
        for pred_name, arity in parsed.get_predicates():
            key = (pred_name, arity)
            if key not in predicates:
                predicates[key] = 0
            predicates[key] += 1
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

    preds = collect_predicates(test_formulas)
    constants = collect_constants(test_formulas)
    print("Collected predicates and their arities:")
    for pred in preds.items():
        (pred_name, arity), occurrence = pred
        print(f"Predicate: {pred_name}, Arity: {arity}, Occurrences: {occurrence}")
    print("\nCollected constants:")
    for const in constants.items():
        const_name, occurrence = const
        print(f"Constant: {const_name}, Occurrences: {occurrence}")

    matrices = create_predicate_terms_matrices(test_formulas)
    predicate_sentences = generate_predicate_embedding_sentences(matrices)
    constant_sentences = generate_constant_embedding_sentences(matrices)

    arity_predicate_matrix = create_predicate_arity_matrix(list(preds.keys()))

    model = SentenceTransformer("all-mpnet-base-v2")
    embedding_predicate_space = build_embedding_space(list(predicate_sentences.values()), lambda x: model.encode(x, normalize_embeddings=False))
    semantic_predicate_matrix_score = build_predicate_similarity_matrix(embedding_predicate_space, predicate_sentences)

    embedding_constant_space = build_embedding_space(list(constant_sentences.values()), lambda x: model.encode(x, normalize_embeddings=False))
    semantic_constant_matrix_score = build_predicate_similarity_matrix(embedding_constant_space, constant_sentences)

    lexical_predicate_matrix_score = build_predicate_lexical_similarity_matrix(list(preds.keys()))
    lexical_constant_matrix_score = build_constant_lexical_similarity_matrix([const for const, occurrence in constants.items()])

    predicate_similarity_score = arity_predicate_matrix * compute_similarity_scores(semantic_predicate_matrix_score, lexical_predicate_matrix_score, alpha=0.1)
    constant_similarity_score = compute_similarity_scores(semantic_constant_matrix_score, lexical_constant_matrix_score, alpha=0.1)

    print("\nPredicate Similarity Scores:")
    print(predicate_similarity_score)

    print("\nConstant Similarity Scores:")
    print(constant_similarity_score)

    unified_predicates = build_unification_map(predicate_similarity_score, preds, threshold=0.8)
    # Print all predicates that have the same representative
    print("\nUnified Predicates:")
    rep_to_members: dict[PRED_KEY, list[PRED_KEY]] = defaultdict(list)
    for pred, rep in unified_predicates.items():
        rep_to_members[rep].append(pred)
    for rep, members in rep_to_members.items():
        if len(members) > 1:
            print(f"Representative: {rep}, Members: {members}")



