import argparse
import os
import shutil
import fitz  # PyMuPDF for PDF extraction
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main(DATA_PATH):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents(DATA_PATH):
    documents = []
    # Load PDF files and preprocess
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DATA_PATH, filename)
            documents.append(preprocess_pdf(file_path))
    return documents

def preprocess_pdf(file_path):
    # Extract text from the PDF file
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()

    # Clean and structure the extracted text
    cleaned_text = clean_text(text)
    structured_data = structure_data(cleaned_text, file_path)

    return Document(page_content=structured_data, metadata={"source": file_path})

def clean_text(text):
    # Remove unnecessary whitespace and artifacts
    cleaned_text = " ".join(text.split())
    return cleaned_text

def structure_data(text, source):
    # Assuming the structure is predictable, we can split the text into sections.
    # Here’s a basic example of how to structure the data
    sections = text.split("Defect Analysis")
    if len(sections) < 2:
        return text  # If the structure is unexpected, return raw text.

    basic_info = sections[0].strip()
    defect_analysis = sections[1].strip().split("Defect Resolution")[0].strip()
    defect_resolution = sections[1].split("Defect Resolution")[1].strip() if "Defect Resolution" in sections[1] else ""

    # Structure it into a format
    structured = f"Basic Information:\n{basic_info}\n\nDefect Analysis:\n{defect_analysis}\n\nDefect Resolution:\n{defect_resolution}"
    return structured

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main(DATA_PATH)
