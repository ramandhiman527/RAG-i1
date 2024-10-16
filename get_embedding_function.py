from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    """
    Use the nomic-embed-text model for embeddings.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
