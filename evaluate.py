import argparse
from query_data import query_rag
import numpy as np
from pprint import pprint
import os
import json

# Define your ground truth dataset here or load from a JSON file
GROUND_TRUTH = [
    {
        "query": "list all date Defect Occurred?",
        "answer": "2023-09-01",
        "relevant_doc_ids": [
            "data/report.pdf:0",
            "data/report.pdf:1"
        ]
    },
    {
        "query": "When did the INS Bagh defect get corrected?",
        "answer": "The INS Bagh defect was corrected on 05/01/24",
        "relevant_doc_ids": [
            "data\Critical Defects Analysis Reports 001.pdf'",
             "data\Critical Defects Analysis Reports 002.pdf",
            "data\Critical Defects Analysis Reports 003.pdf", 
            "data\Critical Defects Analysis Reports 004.pdf"
        ]
    },
    {
        "query": "What is the name of the Ship?",
        "answer": "INS Cheetah",
        "relevant_doc_ids": ["data/report.pdf:0"]
    },
    # Add more queries and their respective ground-truth answers and relevant doc ids
]

# Evaluation metrics
def evaluate_rag_model(ground_truth):
    recall_scores = []
    map_scores = []
    exact_match_scores = []

    total_queries = len(ground_truth)
    print(f"Starting evaluation on {total_queries} queries.")

    for idx, data in enumerate(ground_truth, start=1):
        query = data.get('query')
        true_answer = data.get('answer')
        true_doc_ids = data.get('relevant_doc_ids')

        print(f"\nEvaluating Query {idx}/{total_queries}: '{query}'")

        try:
            # Query the RAG system
            result = query_rag(query)
            retrieved_sources = result.get("sources", [])
            model_answer = result.get("Response", "").strip()

            print(f"Model Answer: {model_answer}")
            print(f"Retrieved Sources: {retrieved_sources}")

            # Evaluate Recall at K
            recall_at_k = recall_at_k_score(retrieved_sources, true_doc_ids, k=5)
            recall_scores.append(recall_at_k)
            print(f"Recall@5: {recall_at_k:.2f}")

            # Evaluate MAP
            map_score = mean_average_precision(retrieved_sources, true_doc_ids)
            map_scores.append(map_score)
            print(f"MAP: {map_score:.2f}")

            # Evaluate Exact Match
            exact_match = exact_match_score(model_answer, true_answer)
            exact_match_scores.append(exact_match)
            print(f"Exact Match: {'Yes' if exact_match else 'No'}")

        except Exception as e:
            print(f"Error evaluating query '{query}': {e}")

    # Calculate overall averages
    avg_recall_at_k = np.mean(recall_scores) if recall_scores else 0
    avg_map = np.mean(map_scores) if map_scores else 0
    avg_exact_match = np.mean(exact_match_scores) if exact_match_scores else 0

    print("\nEvaluation Results:")
    print(f"Average Recall@5: {avg_recall_at_k:.2f}")
    print(f"Average MAP: {avg_map:.2f}")
    print(f"Average Exact Match: {avg_exact_match:.2f}")

    return {
        "Average Recall@5": avg_recall_at_k,
        "Average MAP": avg_map,
        "Average Exact Match": avg_exact_match
    }

def recall_at_k_score(retrieved_sources, true_sources, k=2):
    """
    Calculates Recall at K, checking how many of the relevant documents are in the top K results.
    """
    retrieved_top_k = retrieved_sources[:k]
    relevant_set = set(true_sources)
    retrieved_set = set(retrieved_top_k)
    
    recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set) if relevant_set else 0
    return recall

def mean_average_precision(retrieved_sources, true_sources):
    """
    Calculates the Mean Average Precision (MAP) score.
    """
    relevant_set = set(true_sources)
    if not relevant_set:
        return 0

    ap = 0
    relevant_count = 0
    for i, source in enumerate(retrieved_sources, start=1):
        if source in relevant_set:
            relevant_count += 1
            precision_at_i = relevant_count / i
            ap += precision_at_i

    return ap / len(relevant_set) if len(relevant_set) > 0 else 0

def exact_match_score(model_answer, true_answer):
    """
    Checks if the model's generated answer exactly matches the true answer.
    """
    return 1 if model_answer.lower() == true_answer.lower() else 0

if __name__ == "__main__":
    # Parse command-line arguments if needed
    parser = argparse.ArgumentParser(description="Evaluate RAG model")
    parser.add_argument('--ground_truth', type=str, help="Path to ground truth JSON file", default=None)
    args = parser.parse_args()

    if args.ground_truth:
        if os.path.exists(args.ground_truth):
            with open(args.ground_truth, 'r') as f:
                GROUND_TRUTH = json.load(f)
            print(f"Loaded ground truth data from {args.ground_truth}")
        else:
            print(f"Ground truth file {args.ground_truth} not found.")
            exit(1)
    else:
        print("Using embedded ground truth data.")

    results = evaluate_rag_model(GROUND_TRUTH)
    pprint(results)
