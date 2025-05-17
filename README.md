# Medical Chatbot Application

This Streamlit-based application serves as an AI-powered medical assistant, enabling users to interact with a chatbot for medical queries and upload prescription PDFs for medical history management.

## Features
- **Chatbot Interface**: Engage with a medical chatbot powered by `llama3-med42-8b` using Retrieval-Augmented Generation (RAG).
- **Prescription Upload**: Parse PDF prescriptions with NER to extract and store diagnoses, medications, and notes.
- **Memory System**: Retrieve relevant past conversations using SentenceTransformer embeddings.
- **Intent Classification**: Classify queries with DistilBERT for tailored responses.
- **Medical History Management**: Store and display medical records locally in `medical_history.json`.
- **PDF Generation**: Download prescriptions as PDFs from chatbot responses.
- **Local Persistence**: Save conversation history and settings in JSON and CSV files.

## Key Components
- **NLP Models**:
  - SentenceTransformer (`all-MiniLM-L6-v2`) for embeddings.
  - BERT-based NER (`dslim/bert-base-NER`) for prescription parsing.
  - DistilBERT (`distilbert-base-uncased`) for intent classification.
- **Libraries**: Streamlit, Ollama, pdfplumber, ReportLab, NLTK, Pandas, PyTorch, scikit-learn.
- **Storage**: Local JSON and CSV files with error handling.
- **UI**: Streamlit interface with navigation for Chatbot and Prescription Upload pages.

## Usage
- Navigate to the Chatbot page to ask medical questions or generate prescriptions.
- Upload PDFs on the Prescription Upload page to manage medical history.
- Delete `medical_history.json` manually for data privacy.

## Notes
- **Disclaimer**: This is a demo, not a substitute for professional medical advice.
- **Dependencies**: Install required libraries and set up Ollama.
- **Storage**: Data is stored locally with no external dependencies.
