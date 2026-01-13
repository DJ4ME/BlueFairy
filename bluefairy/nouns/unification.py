import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from bluefairy.grammar.utils import parse_or_false, PRED_KEY
from bluefairy.nouns.embedding import build_embedding_space, build_predicate_similarity_matrix, \
    generate_predicate_embedding_sentences, generate_constant_embedding_sentences
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

    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            continue
        for (pred_name, arity), terms in parsed.get_full_predicates():
            pred_key: PRED_KEY = (pred_name, arity)
            for pos, term in enumerate(terms):
                if is_variable(term):
                    continue
                pred_pos_terms[pred_key][pos].add(term)

    if not pred_pos_terms:
        return []

    predicates = sorted(pred_pos_terms.keys())
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
    for i, matrix in enumerate(matrices):
        print(f"\nPredicate-Terms Matrix for position {i}:")
        print(matrix)

    predicate_sentences = generate_predicate_embedding_sentences(matrices)
    print("\nGenerated embedding sentences for predicates:")
    for key, sentence in predicate_sentences.items():
        print(sentence)

    constant_sentences = generate_constant_embedding_sentences(matrices)
    print("\nGenerated embedding sentences for constants:")
    for key, sentence in constant_sentences.items():
        print(sentence)

    model = SentenceTransformer("all-mpnet-base-v2")
    embedding_space = build_embedding_space(list(predicate_sentences.values()), lambda x: model.encode(x, normalize_embeddings=False))
    semantic_predicate_matrix_score = build_predicate_similarity_matrix(embedding_space, predicate_sentences)
    print("\nPredicate Semantic Similarity Matrix:")
    print(semantic_predicate_matrix_score)

    embedding_space = build_embedding_space(list(constant_sentences.values()), lambda x: model.encode(x, normalize_embeddings=False))
    semantic_constant_matrix_score = build_predicate_similarity_matrix(embedding_space, constant_sentences)
    print("\nConstant Semantic Similarity Matrix:")
    print(semantic_constant_matrix_score)

    lexical_predicate_matrix_score = build_predicate_lexical_similarity_matrix(list(preds.keys()))
    print("\nPredicate Lexical Similarity Matrix:")
    print(lexical_predicate_matrix_score)

    lexical_constant_matrix_score = build_constant_lexical_similarity_matrix([const for const, occurrence in constants.items()])
    print("\nConstant Lexical Similarity Matrix:")
    print(lexical_constant_matrix_score)



