from typing import Callable
import numpy as np
import pandas as pd

from bluefairy.grammar.utils import PRED_KEY


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


def build_predicate_similarity_matrix(
    embedding_space: dict[str, object],
    predicate_to_sentence: dict[PRED_KEY, str],
) -> pd.DataFrame:
    sentences: list[str] = embedding_space["sentences"]
    embeddings: np.ndarray = embedding_space["embeddings"]

    sentence_to_index = {s: i for i, s in enumerate(sentences)}

    predicates = list(predicate_to_sentence.keys())
    n = len(predicates)

    sim_matrix = np.zeros((n, n), dtype=np.float32)

    pred_indices = []
    for pred in predicates:
        sentence = predicate_to_sentence[pred]
        if sentence not in sentence_to_index:
            raise ValueError(f"Sentence for predicate {pred} not found in embedding space")
        pred_indices.append(sentence_to_index[sentence])

    for i in range(n):
        ei = embeddings[pred_indices[i]]
        for j in range(i, n):
            score = float(np.dot(ei, embeddings[pred_indices[j]]))
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    return pd.DataFrame(sim_matrix, index=predicates, columns=predicates)

