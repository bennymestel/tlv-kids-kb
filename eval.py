"""Score search quality against data/eval_queries.json.

Usage: python eval.py                            # tuning set
       python eval.py data/eval_heldout.json      # held-out set: check rarely, never tune to it
       python eval.py data/eval_logged.json       # real queries from the usage log

For each query, finds the rank of the first result whose question is in the
query's "expected" list, then reports per group and overall:
  hit@3 - fraction of queries with a correct result in the top 3
          (an empty "expected" list counts as a hit only if nothing is shown)
  MRR   - average of 1/rank (0 if not shown)
  junk  - average number of shown results that are not in "expected"
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from search import load, search

EVAL_PATH = Path("data/eval_queries.json")
K = 5


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_PATH
    cases = json.loads(path.read_text())
    index, qas = load()
    known_questions = {qa["question"] for qa in qas}

    by_group = defaultdict(list)
    junk_by_group = defaultdict(list)
    misses = []
    for case in cases:
        for q in case["expected"]:
            if q not in known_questions:
                print(f"WARNING: expected question not in qa.json: {q!r}")
        if not case["expected"]:
            pattern = re.compile(rf"\b{re.escape(case['query'])}", re.IGNORECASE)
            for q in known_questions:
                if pattern.search(q):
                    print(f"WARNING: {case['query']!r} expects no results, but data has: {q!r}")

        results = search(index, qas, case["query"], case["category"], k=K)
        if case["expected"]:
            rank = next(
                (i + 1 for i, qa in enumerate(results) if qa["question"] in case["expected"]),
                None,
            )
        else:
            rank = None if results else 1
        by_group[case["group"]].append(rank)
        junk_by_group[case["group"]].append(
            sum(1 for qa in results if qa["question"] not in case["expected"])
        )
        if rank is None or rank > 3:
            top = f"{len(results)} shown, first: {results[0]['question']}" if results else "(no results)"
            misses.append((case["query"], rank, top))

    def score(ranks):
        hit3 = sum(1 for r in ranks if r and r <= 3) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        return hit3, mrr

    print(f"\n{'group':<16}{'n':>4}{'hit@3':>8}{'MRR':>8}{'junk':>8}")
    all_ranks, all_junk = [], []
    for group, ranks in by_group.items():
        hit3, mrr = score(ranks)
        junk = junk_by_group[group]
        print(f"{group:<16}{len(ranks):>4}{hit3:>8.2f}{mrr:>8.2f}{sum(junk) / len(junk):>8.1f}")
        all_ranks.extend(ranks)
        all_junk.extend(junk)
    hit3, mrr = score(all_ranks)
    print(f"{'overall':<16}{len(all_ranks):>4}{hit3:>8.2f}{mrr:>8.2f}{sum(all_junk) / len(all_junk):>8.1f}")

    if misses:
        print(f"\nMissed top 3 ({len(misses)}):")
        for query, rank, top in misses:
            where = f"rank {rank}" if rank else f"not in top {K}"
            print(f"  {query!r}: {where} | top result: {top}")


if __name__ == "__main__":
    main()
