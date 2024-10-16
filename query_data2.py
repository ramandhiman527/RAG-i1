from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import CharacterTextSplitter
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Few-shot Prompt Template
FEW_SHOT_PROMPT_TEMPLATE = """
You are an intelligent assistant that answers questions based on the given context from a set of documents. Use the provided examples to format your answers. If the question cannot be answered based on the context, respond with "I don't know."

### Examples:

Example 1:
Context: 
"The report states that the ship encountered an electrical failure on 01/01/2024, and the issue was resolved on 05/01/2024 by replacing the Magnetron."
Question: When did the defect occur?
Answer: The defect occurred on 01/01/2024.

Example 2:
Context: 
"The repair of System 1 was successfully completed by Ram Vilas from Repair Unit 1 on 07/07/2024."
Question: Who resolved the defect?
Answer: The defect was resolved by Ram Vilas from Repair Unit 1.

Example 3:
Context:
"The defective engine component was identified as the primary cause of the malfunction, and replacement parts were ordered on 10/10/2023."
Question: When were the replacement parts ordered?
Answer: The replacement parts were ordered on 10/10/2023.

Example 4:
Context: 
"The engineers concluded that the problem was due to a software glitch affecting the cooling system. A patch was released on 11/11/2023 to fix the issue."
Question: What caused the issue?
Answer: The issue was caused by a software glitch affecting the cooling system.

### Now answer the following question based on the context provided:

Context: {context}

Question: {question}

Answer:
"""



def main():
    query_text = "When did the defect occur?"
    print("Query text: ", query_text)
    data = query_rag(query_text)
    return data


def query_rag(query_text: str):
    # Step 1: Prepare the context
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Step 2: Retrieve documents based on context (base retrieval)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    context_docs = retriever.invoke(query_text)  # Updated to use 'invoke' instead of 'get_relevant_documents'

    # Step 3: Contextual compression with embeddings filter
    embeddings_filter = EmbeddingsFilter(embeddings=embedding_function, similarity_threshold=0.75)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    
    # Retrieve relevant documents based on the compressed context
    compressed_docs = compression_retriever.invoke(query_text)  # Updated to use 'invoke'

    # Step 4: Retrieve documents based on the query (direct retrieval)
    query_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    query_docs = query_retriever.invoke(query_text)  # Updated to use 'invoke'

    # Step 5: Intersect the retrieved document IDs
    context_doc_ids = {doc.metadata["id"] for doc in context_docs}
    query_doc_ids = {doc.metadata["id"] for doc in query_docs}
    intersected_doc_ids = context_doc_ids.intersection(query_doc_ids)

    # Step 6: Re-rank the relevant documents based on the query
    final_retrieved_docs = [
        doc for doc in compressed_docs if doc.metadata["id"] in intersected_doc_ids
    ]

    # Prepare the context from the retrieved and compressed documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_retrieved_docs])
    
    # Dynamic few-shot prompting with context
    prompt_template = ChatPromptTemplate.from_template(FEW_SHOT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Step 7: Use Ollama model for generating a response
    model = Ollama(model="phi3.5")
    response_text = model.invoke(prompt)

    # Prepare the sources for output
    sources = [doc.metadata.get("id", None) for doc in final_retrieved_docs]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    data = {
        "query_text": query_text,
        "sources": sources,
        "Response": response_text
    }
    return data



if __name__ == "__main__":
    main()
