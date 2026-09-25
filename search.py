"""Load data/qa.json into a hybrid search index and search it.

Two searches run on every query and their rankings are combined:
  - embeddings (Chroma) over the question only: good at meaning/paraphrases
  - BM25 over character trigrams of the question + answers: good at exact names
    and transliterated Hebrew with spelling variants (moked/mokad, mosach/musach)
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from query_terms import expand

QA_PATH = Path("data/qa.json")
# Question embeddings saved so the app doesn't recompute them on every start (the slow part).
# Stored with the questions they came from; if qa.json changes, they're recomputed and resaved.
EMBEDDINGS_PATH = Path("data/question_embeddings.npz")

# How many results each search contributes before fusion. Kept small: with long
# lists, weak results that appear low in *both* lists outscore a result that one
# search ranks #1 (e.g. an exact name match), which dropped eval hit@3 from 0.84 to 0.65.
CANDIDATES = 10
RRF_K = 60  # standard reciprocal rank fusion constant
# Drop results much weaker than the best one for the same query. Relative, not absolute:
# absolute scores didn't separate right from wrong (some correct matches scored worse than
# the best match for an unrelated query). Tuned on eval_queries.json: ~10 -> ~4 results shown.
MAX_DISTANCE_GAP = 0.15  # keep if embedding distance is within this of the best result's
MIN_KEYWORD_SHARE = 0.6  # or if keyword score is at least this share of the best result's
# Show nothing if no search word appears in the data and no question is close in meaning.
MIN_WORD_COVERAGE = 0.5
NO_MATCH_DISTANCE = 0.6
STOPWORDS = set(
    "the and for with any anyone someone somewhere good best who where what how get can "
    "does are you your our their this that from near need looking recommend recommendation".split()
)


@dataclass
class Index:
    collection: chromadb.Collection
    bm25: BM25Okapi | None


def trigrams(text: str) -> list[str]:
    """'moked' -> [' mo', 'mok', 'oke', 'ked', 'ed ']. Padding with spaces lets
    word starts/ends count, so short words still produce useful trigrams."""
    grams = []
    for word in re.findall(r"\w+", text.lower()):
        padded = f" {word} "
        grams.extend(padded[i:i + 3] for i in range(len(padded) - 2))
    return grams


def word_coverage(index: Index, word: str) -> float:
    """Largest share of the word's trigrams found together in one Q&A entry."""
    grams = set(trigrams(word))
    return max(len(grams & doc.keys()) for doc in index.bm25.doc_freqs) / len(grams)


def question_embeddings(questions: list[str], ef) -> np.ndarray:
    if EMBEDDINGS_PATH.exists():
        saved = np.load(EMBEDDINGS_PATH)
        if saved["questions"].tolist() == questions:
            return saved["embeddings"]
    embeddings = np.array(ef(questions), dtype=np.float32)
    np.savez(EMBEDDINGS_PATH, questions=np.array(questions), embeddings=embeddings)
    return embeddings


def load():
    qas = json.loads(QA_PATH.read_text())

    client = chromadb.Client()
    ef = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    collection = client.create_collection("qa_pairs", embedding_function=ef)

    bm25 = None
    if qas:
        questions = [qa["question"] for qa in qas]
        collection.add(
            ids=[str(i) for i in range(len(qas))],
            documents=questions,
            embeddings=question_embeddings(questions, ef),
            metadatas=[{"category": qa["category"]} for qa in qas],
        )
        bm25 = BM25Okapi([
            trigrams(qa["question"] + " " + " ".join(
                " ".join(filter(None, [a["text"], a.get("name")])) for a in qa["answers"]
            ))
            for qa in qas
        ])
    return Index(collection, bm25), qas


def search(index: Index, qas, query: str, category: str | None = None, k: int = 10):
    allowed = [i for i, qa in enumerate(qas) if not category or qa["category"] == category]

    if not query:
        return [qas[i] for i in allowed[:k]]
    if not allowed:
        return []

    query = expand(query)
    where = {"category": category} if category else None
    res = index.collection.query(
        query_texts=[query], n_results=min(CANDIDATES, len(allowed)), where=where
    )
    embedding_ranked = [int(i) for i in res["ids"][0]]
    distance = dict(zip(embedding_ranked, res["distances"][0]))
    best_distance = min(distance.values())

    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3 and w not in STOPWORDS]
    if best_distance > NO_MATCH_DISTANCE and not any(
        word_coverage(index, w) >= MIN_WORD_COVERAGE for w in words
    ):
        return []

    scores = index.bm25.get_scores(trigrams(query))
    best_score = max(scores[i] for i in allowed)
    keyword_ranked = sorted((i for i in allowed if scores[i] > 0), key=lambda i: -scores[i])
    keyword_ranked = keyword_ranked[:CANDIDATES]

    # Reciprocal rank fusion: each list gives 1/(RRF_K + rank); sum across lists.
    fused: dict[int, float] = {}
    for ranked in (embedding_ranked, keyword_ranked):
        for rank, i in enumerate(ranked, start=1):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)

    def strong(i):
        close = i in distance and distance[i] - best_distance <= MAX_DISTANCE_GAP
        return close or (best_score > 0 and scores[i] >= MIN_KEYWORD_SHARE * best_score)

    best = [i for i in sorted(fused, key=lambda i: -fused[i]) if strong(i)][:k]
    return [qas[i] for i in best]
