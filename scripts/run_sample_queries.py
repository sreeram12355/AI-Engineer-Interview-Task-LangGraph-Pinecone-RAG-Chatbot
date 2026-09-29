"""Runs the six validation queries and writes sample_outputs.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph import ask  # noqa: E402

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",  # out-of-scope: must be refused
]

if __name__ == "__main__":
    results = []
    for q in QUERIES:
        r = ask(q)
        print(f"\nQ: {q}\nA: {r['final_answer']}\nconfidence={r['confidence_score']} "
              f"chunks={len(r['retrieved_context_chunks'])}")
        results.append(r)
    with open("sample_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved sample_outputs.json")
