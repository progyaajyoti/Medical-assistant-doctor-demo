# Medical-assistant-doctor-demo
This Streamlit-based application serves as an AI-powered medical assistant, enabling users to interact with a chatbot for medical queries and upload prescription PDFs for medical history management. It integrates advanced NLP models and local storage for conversation and medical history.
## Features
-Chatbot Interface: Users can ask medical questions or describe symptoms, with responses generated using the llama3-med42-8b model via Ollama, enhanced by Retrieval-Augmented Generation (RAG).
-Prescription Upload: Upload PDF prescriptions, which are parsed using NER (Named Entity Recognition) to extract diagnoses, medications, dates, and notes, stored in a local JSON file.
-Memory System: Utilizes a SentenceTransformer model (all-MiniLM-L6-v2) for embedding-based retrieval of relevant past conversations, with a configurable top_k for context inclusion.
-Intent Classification: Employs DistilBERT to classify user queries into memory, non-memory, or medical history intents, tailoring responses accordingly.
-Medical History Management: Stores and displays parsed prescription data locally in medical_history.json, with a user-friendly interface to view records.
-PDF Generation: Generates downloadable PDF prescriptions from chatbot responses when requested.
-Local Persistence: Saves conversation history (conversation_history.json) and top_k settings (top_k_history.csv) locally for session continuity.

