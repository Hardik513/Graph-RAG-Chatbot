# Graph RAG Chatbot with Auto-Updating Memory

This project implements a Retrieval-Augmented Generation (RAG) chatbot that uses a dynamic Knowledge Graph as its memory. It extracts facts from user statements using `spaCy`, stores them in a `NetworkX` directed graph, and uses a local Hugging Face Transformer (`google/flan-t5-base`) to answer user queries based on the stored graph context. It also supports independent memory for multiple users.

## ⚙️ Setup Instructions

1. **Install Dependencies**
   Make sure you have Python installed, then run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
Download the NLP Model
The chatbot uses spaCy for grammatical parsing. Download the English model by running:

    ```bash
    python -m spacy download en_core_web_sm
Run the Application
Start the Streamlit interface:

    ```bash
      streamlit run app.py
## Example Inputs (Graph Updates)
Type these statements into the chat (without question marks) to populate the Knowledge Graph:

I am Alex

I live in Goa

I have a dog

My dog name is Gon

Gon loves pizza

Emma studies biology

Emma is a student

Biology is a science

Microsoft employs Emma

Microsoft created Windows

## Example Queries (RAG Retrieval)
Ask these questions (ensuring you include the ?) to test the chatbot's memory:

Who am I?

Where do I live?

What do I have?

What does Gon love?

Who employs Emma?

## Graph Visualization
See the attached screenshot graph_screenshot.png in this repository to view the live NetworkX multi-user visualization generated within the Streamlit UI



