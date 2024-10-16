# rerank_documents.py

import logging
import subprocess
import shlex
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_ollama(model_name='mxbai-embed-large'):
    """
    Initialize the Ollama model for document reranking.
    
    Args:
        model_name (str): The name of the Ollama model to use.
        
    Returns:
        bool: True if initialization is successful, False otherwise.
    """
    try:
        # Test if the model is accessible by running a simple command
        test_prompt = "Test relevance scoring."
        command = f"ollama run {model_name} \"{test_prompt}\""
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            logging.error(f"Failed to initialize Ollama model '{model_name}': {result.stderr.strip()}")
            return False
        
        logging.info(f"Ollama model '{model_name}' initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Exception during Ollama model initialization: {e}")
        return False

def get_relevance_score_ollama(query, document, model_name='mxbai-embed-large'):
    """
    Get the relevance score of a document to a query using the Ollama model.
    
    Args:
        query (str): The user's query.
        document (str): The document content.
        model_name (str): The name of the Ollama model to use.
        
    Returns:
        float: The relevance score (e.g., between 0 and 5). Returns 0 on failure.
    """
    try:
        # Define the prompt to get a relevance score
        prompt = f"Rate the relevance of the following document to the query on a scale of 1 to 5.\n\nQuery: {query}\n\nDocument: {document}\n\nScore:"
        
        # Execute the Ollama command
        command = f"ollama run {model_name} \"{prompt}\""
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logging.error(f"Ollama model error: {result.stderr.strip()}")
            return 0.0
        
        # Parse the score from the model's output
        score_str = result.stdout.strip()
        
        # Attempt to extract a float from the output
        try:
            score = float(score_str)
            # Clamp the score between 1 and 5
            score = max(1.0, min(5.0, score))
            return score
        except ValueError:
            logging.error(f"Invalid score format received from Ollama: '{score_str}'")
            return 0.0
    
    except subprocess.TimeoutExpired:
        logging.error("Ollama model request timed out.")
        return 0.0
    except Exception as e:
        logging.error(f"Exception during relevance scoring: {e}")
        return 0.0

def rerank_documents(query, doc_ids, chroma_db_path, model_name='mxbai-embed-large'):
    """
    Rerank documents based on their relevance to the query using the Ollama model.
    
    Args:
        query (str): The user's query.
        doc_ids (list): List of document IDs to rerank.
        chroma_db_path (str): Path to the Chroma database.
        model_name (str): The name of the Ollama model to use.
        
    Returns:
        list: List of reranked document IDs sorted by relevance (highest first).
    """
    # Initialize Ollama model
    if not initialize_ollama(model_name):
        logging.error("Ollama model initialization failed. Returning unsorted document IDs.")
        return doc_ids  # Return original order if initialization fails
    
    try:
        # Initialize Chroma client
        db = Chroma(persist_directory=chroma_db_path, embedding_function=get_embedding_function())
        documents = db.get(include=["documents"], where={"id": doc_ids})
        
        # Extract document contents
        docs_contents = [doc.page_content for doc in documents["documents"]]
        
        # Get relevance scores for each document
        scores = []
        for doc_content in docs_contents:
            score = get_relevance_score_ollama(query, doc_content, model_name)
            scores.append(score)
        
        # Combine document IDs with their scores
        doc_scores = list(zip(doc_ids, scores))
        
        # Sort documents based on scores in descending order
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Extract sorted document IDs
        sorted_doc_ids = [doc_id for doc_id, score in sorted_docs]
        
        logging.info("Documents reranked successfully using Ollama.")
        return sorted_doc_ids
    
    except Exception as e:
        logging.error(f"Error during document reranking: {e}")
        return doc_ids  # Return original order on failure

if __name__ == "__main__":
    # Example usage
    query_example = "What are the main components of the neural network?"
    doc_ids_example = ["doc1", "doc2", "doc3"]
    chroma_db_path_example = "data/chroma"  # Update this path as needed
    model_name_example = "mxbai-embed-large"  # Replace with your actual model name
    
    reranked_docs = rerank_documents(query_example, doc_ids_example, chroma_db_path_example, model_name_example)
    print("Reranked Document IDs:", reranked_docs)
