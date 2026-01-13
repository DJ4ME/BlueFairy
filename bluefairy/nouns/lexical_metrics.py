import pandas as pd
from difflib import SequenceMatcher
from bluefairy.grammar.utils import PRED_KEY


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_predicate_lexical_similarity_matrix(
    predicates: list[PRED_KEY],
) -> pd.DataFrame:
    n = len(predicates)

    sim_matrix = pd.DataFrame(
        0.0,
        index=predicates,
        columns=predicates,
        dtype=float,
    )

    for i in range(n):
        name_i, _ = predicates[i]
        for j in range(i, n):
            name_j, _ = predicates[j]
            score = lexical_similarity(name_i, name_j)
            sim_matrix.iat[i, j] = score
            sim_matrix.iat[j, i] = score

    return sim_matrix


def build_constant_lexical_similarity_matrix(
    constants: list[str],
) -> pd.DataFrame:
    index = pd.Index(constants, name="constant")

    sim_matrix = pd.DataFrame(
        0.0,
        index=index,
        columns=index,
        dtype=float,
    )

    n = len(constants)
    for i in range(n):
        for j in range(i, n):
            score = lexical_similarity(constants[i], constants[j])
            sim_matrix.iat[i, j] = score
            sim_matrix.iat[j, i] = score

    return sim_matrix

