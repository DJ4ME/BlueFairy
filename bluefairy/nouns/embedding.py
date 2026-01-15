import numpy as np
import pandas as pd
from typing import Callable
from bluefairy.grammar.utils import PRED_KEY


def generate_predicate_embedding_sentences(
        df_list: list[pd.DataFrame]
) -> dict[PRED_KEY,str]:
    """
    Generate canonical sentences for embeddings from a list of predicate-term DataFrames.
    Each DataFrame corresponds to one argument position (position index = df_list index).
    Each row = predicate (name, arity), each column = term, values = 1/0.

    :param df_list: the list of DataFrames representing predicate terms matrices.
    :return: a list of sentences describing the predicates and their related terms.
    """
    if not df_list:
        return []

    predicates = df_list[0].index.tolist()

    sentences = {}

    for pred_key in predicates:
        pred_name, arity = pred_key
        parts = []
        for pos_idx in range(arity):
            if pos_idx >= len(df_list):
                parts.append("a logic variable")
                continue

            df = df_list[pos_idx]
            # TODO: I suppose this is not the best way efficiency-wise.
            # Consider using a multi-index DataFrame in the future.
            row = df.iloc[df.index.get_loc(pred_key)]
            args = [col for col, val in row.items() if val == 1]
            if args:
                parts.append("something like " + ", ".join([f"'{x}'" for x in args]))
            else:
                parts.append("a logic variable")

        sentence = f"Predicate '{pred_name}' relates "
        if len(parts) == 1:
            sentence += parts[0]
        else:
            sentence += " to ".join(parts) + "."
        sentences[pred_key] = sentence

    return sentences


def generate_constant_embedding_sentences(
    df_list: list[pd.DataFrame],
) -> dict[str, str]:
    constant_to_sentence: dict[str, str] = {}

    constants = df_list[0].columns

    for constant in constants:
        parts = []

        for pos, df in enumerate(df_list, start=1):
            used_preds = df.index[df[constant] > 0]

            if len(used_preds) == 0:
                continue

            pred_names = sorted({pred_name for pred_name, _ in used_preds})

            if len(pred_names) == 1:
                part = (
                    f"as argument {pos} of predicate '{pred_names[0]}'"
                )
            else:
                preds = ", ".join(f"'{p}'" for p in pred_names)
                part = (
                    f"as argument {pos} of predicates {preds}"
                )

            parts.append(part)

        if not parts:
            continue

        sentence = (
            f"Constant '{constant}' is used "
            + ", ".join(parts)
            + "."
        )

        constant_to_sentence[constant] = sentence

    return constant_to_sentence


def build_embedding_space(
    sentences: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, object]:
    embeddings = embed_fn(sentences)

    if len(embeddings) != len(sentences):
        raise ValueError("Number of embeddings does not match number of sentences")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embeddings = embeddings / norms

    return {
        "sentences": sentences,
        "embeddings": embeddings,
    }


def embedding_similarity(
    embedding_space: dict[str, object],
    idx1: int,
    idx2: int,
) -> float:
    embeddings: np.ndarray = embedding_space["embeddings"]

    if idx1 >= len(embeddings) or idx2 >= len(embeddings):
        raise IndexError("Embedding index out of range")

    return float(np.dot(embeddings[idx1], embeddings[idx2]))


def embedding_similarity_by_sentence(
    embedding_space: dict[str, object],
    sentence1: str,
    sentence2: str,
) -> float:
    sentences: list[str] = embedding_space["sentences"]

    idx1 = sentences.index(sentence1)
    idx2 = sentences.index(sentence2)

    return embedding_similarity(embedding_space, idx1, idx2)


def build_similarity_matrix(
    embedding_space: dict[str, object],
    mapping: dict[str or PRED_KEY, str],
) -> pd.DataFrame:
    sentences: list[str] = embedding_space["sentences"]
    embeddings: np.ndarray = embedding_space["embeddings"]

    sentence_to_index = {s: i for i, s in enumerate(sentences)}

    elements = list(mapping.keys())
    n = len(elements)

    sim_matrix = np.zeros((n, n), dtype=np.float32)

    indices = []
    for element in elements:
        sentence = mapping[element]
        if sentence not in sentence_to_index:
            raise ValueError(f"Sentence for {'constant' if isinstance(mapping.keys()[0], str) else 'predicate'} {element} not found in embedding space")
        indices.append(sentence_to_index[sentence])

    for i in range(n):
        ei = embeddings[indices[i]]
        for j in range(i, n):
            score = float(np.dot(ei, embeddings[indices[j]]))
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    return pd.DataFrame(sim_matrix, index=elements, columns=elements)



def build_predicate_similarity_matrix(
    embedding_space: dict[str, object],
    predicate_to_sentence: dict[PRED_KEY, str],
) -> pd.DataFrame:
    return build_similarity_matrix(embedding_space, predicate_to_sentence)


def build_constant_similarity_matrix(
    embedding_space: dict[str, object],
    constant_to_sentence: dict[str, str],
) -> pd.DataFrame:
    return build_similarity_matrix(embedding_space, constant_to_sentence)
