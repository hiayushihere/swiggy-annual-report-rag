import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.retriever import retrieve

EVAL_FILE = ROOT / "data/eval/questions_retrieval.json"


def check_hit(expected_page, hits):
    """
    expected_page = int or None
    hits = list of retrieved chunk dicts
    """
    retrieved_pages = [h["meta"].get("page") for h in hits]

    if expected_page is None:
        return expected_page not in retrieved_pages

    return expected_page in retrieved_pages


def evaluate():
    if not EVAL_FILE.exists():
        print(" Evaluation file missing:", EVAL_FILE)
        return

    tests = json.load(open(EVAL_FILE))

    total = len(tests)
    score = 0

    print("\n Running Retrieval Accuracy Evaluation\n")

    for t in tests:
        qid = t["id"]
        q = t["question"]
        expected_page = t["expected_page"]

        print(f"→ [{qid}] {q}")

        hits = retrieve(q, topk=10, rerank_topk=5)

        ok = check_hit(expected_page, hits)
        retrieved_pages = list({h["meta"].get("page") for h in hits})

        if ok:
            score += 1
            print(f"   ✓ PASS  (expected={expected_page}, retrieved={retrieved_pages})")
        else:
            print(f"   ✗ FAIL  (expected={expected_page}, retrieved={retrieved_pages})")

        print()

    print(f"Final Retrieval Accuracy: {score}/{total}")


if __name__ == "__main__":
    evaluate()
