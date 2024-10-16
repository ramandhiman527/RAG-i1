import argparse
import os
import gradio as gr
import shutil
from populate_database import load_documents, split_documents, add_to_chroma, clear_database
from query_data import query_rag

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def reset_database():
    print("✨ Clearing Database")
    clear_database()

def train_model(file_paths):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # Move uploaded files to DATA_PATH
    for file_path in file_paths:
        try:
            destination_path = os.path.join(DATA_PATH, os.path.basename(file_path))
            shutil.copy(file_path, destination_path)
            print(f"Copied {file_path} to {destination_path}")
        except Exception as e:
            print(f"Error copying file {file_path}: {e}")
            return f"Error copying file {file_path}: {e}"
    
    # Load, split, and add to Chroma
    try:
        documents = load_documents(DATA_PATH)
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        return "Training completed successfully."
    except Exception as e:
        return f"An error occurred during training: {e}"

def test_model(query):
    try:
        result = query_rag(query)
        response_text = result.get("Response", "No response generated.")
        sources = result.get("sources", [])
        formatted_response = f"Response: {response_text}\n\nSources: {sources}"
        return formatted_response
    except Exception as e:
        return f"An error occurred during testing: {e}"

# Define the train interface
with gr.Blocks() as train_interface:
    gr.Markdown("# Train Model")
    with gr.Row():
        with gr.Column():
            file_upload = gr.File(
                label="Upload Training Data",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )
            train_button = gr.Button("Train")
        with gr.Column():
            train_response = gr.Textbox(label="Training Response", lines=5)
    
    train_button.click(train_model, inputs=file_upload, outputs=train_response)

# Define the test interface
with gr.Blocks() as test_interface:
    gr.Markdown("# Test Model")
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Enter Query", lines=2)
            test_button = gr.Button("Test")
        with gr.Column():
            test_response = gr.Textbox(label="Test Response", lines=5)
    
    test_button.click(test_model, inputs=query_input, outputs=test_response)

# Combine the interfaces into a single app
app = gr.TabbedInterface([train_interface, test_interface], ["Train", "Test"])

# Launch the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System Interface")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        reset_database()

    app.launch()
