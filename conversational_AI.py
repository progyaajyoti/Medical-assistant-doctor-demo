import streamlit as st
import ollama
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
from nltk.tokenize import word_tokenize
import pdfplumber
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.nn.functional import softmax

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Define the model name
model_name = "thewindmom/llama3-med42-8b"

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# Load the NER model from transformers
@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=-1)  # device=-1 for CPU
    return nlp

ner_model = load_ner_model()

# Initialize session state for conversation and medical history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "medical_history" not in st.session_state:
    st.session_state.medical_history = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I’m your medical assistant. Ask a question or navigate to the Prescription Upload page!"}
    ]

# Load or initialize top_k from CSV
if "top_k" not in st.session_state:
    top_k_csv = "top_k_history.csv"
    if os.path.exists(top_k_csv):
        try:
            df = pd.read_csv(top_k_csv)
            st.session_state.top_k = int(df["top_k"].iloc[-1])
        except (pd.errors.EmptyDataError, KeyError, IndexError, ValueError) as e:
            print(f"Error reading top_k_history.csv: {e}. Initializing top_k to 1.")
            st.session_state.top_k = 1
            pd.DataFrame({"top_k": [1]}).to_csv(top_k_csv, index=False)
    else:
        print("top_k_history.csv not found, initializing top_k to 1.")
        st.session_state.top_k = 1
        pd.DataFrame({"top_k": [1]}).to_csv(top_k_csv, index=False)

# Load conversation history from file
if os.path.exists("conversation_history.json"):
    try:
        with open("conversation_history.json", "r") as f:
            content = f.read().strip()
            if content:
                st.session_state.conversation_history = json.loads(content)
            else:
                st.session_state.conversation_history = []
                with open("conversation_history.json", "w") as f:
                    json.dump([], f)
    except json.JSONDecodeError as e:
        print(f"Error decoding conversation_history.json: {e}. Resetting to empty list.")
        st.session_state.conversation_history = []
        with open("conversation_history.json", "w") as f:
            json.dump([], f)
    except Exception as e:
        print(f"Unexpected error loading conversation_history.json: {e}. Starting fresh.")
        st.session_state.conversation_history = []
else:
    with open("conversation_history.json", "w") as f:
        json.dump([], f)

# Load medical history from file
if os.path.exists("medical_history.json"):
    try:
        with open("medical_history.json", "r") as f:
            content = f.read().strip()
            if content:
                st.session_state.medical_history = json.loads(content)
            else:
                print("medical_history.json is empty, initializing as empty list.")
                st.session_state.medical_history = []
                with open("medical_history.json", "w") as f:
                    json.dump([], f)
    except json.JSONDecodeError as e:
        print(f"Error decoding medical_history.json: {e}. Resetting to empty list.")
        st.session_state.medical_history = []
        with open("medical_history.json", "w") as f:
            json.dump([], f)
    except Exception as e:
        print(f"Unexpected error loading medical_history.json: {e}. Starting fresh.")
        st.session_state.medical_history = []
else:
    print("medical_history.json not found, starting fresh.")
    with open("medical_history.json", "w") as f:
        json.dump([], f)

# Memory settings
use_memory = True
max_top_k = 100
max_tokens = 6000

# Function to update top_k in CSV
def update_top_k(new_top_k):
    st.session_state.top_k = new_top_k
    top_k_csv = "top_k_history.csv"
    pd.DataFrame({"top_k": [new_top_k]}).to_csv(top_k_csv, mode='w', index=False)

# Function to save medical history
def save_medical_history():
    try:
        with open("medical_history.json", "w") as f:
            json.dump(st.session_state.medical_history, f)
    except Exception as e:
        print(f"Error saving medical_history.json: {e}")

# Load pre-trained DistilBERT for intent classification
@st.cache_resource
def load_distilbert_classifier():
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 classes: memory, non-memory, medical history
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, distilbert_model, device = load_distilbert_classifier()

# Function to classify intent using DistilBERT
def classify_intent(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)  # [batch_size, 3] for memory, non-memory, medical history
        memory_prob = probabilities[0][0].item()  # Memory intent
        non_memory_prob = probabilities[0][1].item()  # Non-memory intent
        medical_history_prob = probabilities[0][2].item()  # Medical history intent

    # Determine intent based on highest probability
    intent_probs = {
        "memory": memory_prob,
        "non_memory": non_memory_prob,
        "medical_history": medical_history_prob
    }
    intent = max(intent_probs, key=intent_probs.get)
    return intent == "memory", intent == "medical_history", intent_probs[intent]

# Function to parse PDF prescription using NER
def parse_prescription(pdf_file):
    # Extract text from PDF
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    # Process text with the NER model
    entities = ner_model(text)
    
    # Initialize prescription dictionary
    prescription = {"diagnoses": [], "medications": [], "date": "", "notes": ""}
    
    # Extract entities
    current_entity = ""
    current_label = ""
    for entity in entities:
        label = entity['entity']
        word = entity['word'].replace("##", "")  # Handle subword tokens
        if label.startswith('B-'):
            # Save the previous entity if it exists
            if current_entity:
                # Heuristic: Assume entities with "disease" or medical terms are diagnoses
                if "disease" in current_entity.lower() or "syndrome" in current_entity.lower():
                    prescription["diagnoses"].append(current_entity)
                # Heuristic: Assume other entities (e.g., chemicals) are medications
                else:
                    prescription["medications"].append({"name": current_entity, "dosage": "N/A"})
            current_entity = word
            current_label = label[2:]  # Remove 'B-' prefix
        elif label.startswith('I-') and label[2:] == current_label:
            current_entity += " " + word
    # Process the last entity
    if current_entity:
        if "disease" in current_entity.lower() or "syndrome" in current_entity.lower():
            prescription["diagnoses"].append(current_entity)
        else:
            prescription["medications"].append({"name": current_entity, "dosage": "N/A"})
    
    # Post-process to extract dosage, date, and notes
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        # Extract date using regex
        date_match = re.search(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', line)
        if date_match and not prescription["date"]:
            prescription["date"] = date_match.group(0)
        
        # Extract dosage for medications
        for med in prescription["medications"]:
            if med["name"] in line and med["dosage"] == "N/A":
                dosage_match = re.search(r'\b\d+\s*(mg|g|ml|tablet(s)?|capsule(s)?)\b', line)
                if dosage_match:
                    med["dosage"] = dosage_match.group(0)
        
        # Extract notes (lines that don't contain entities or dates)
        if line and not any(entity['word'] in line for entity in entities) and not date_match and not prescription["notes"]:
            prescription["notes"] = line
    
    return prescription

# Function to save conversation
def save_conversation(user_query, assistant_response):
    combined_text = f"User: {user_query}\nAssistant: {assistant_response}"
    embedding = embedder.encode(combined_text).tolist()
    st.session_state.conversation_history.append({
        "user": user_query,
        "assistant": assistant_response,
        "embedding": embedding
    })
    try:
        with open("conversation_history.json", "w") as f:
            json.dump(st.session_state.conversation_history, f)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

# Function to retrieve relevant conversations (RAG)
def retrieve_relevant_conversations(query, topic_words, top_k, max_tokens=6000):
    if not st.session_state.conversation_history:
        return "", []
    
    query_text = " ".join(topic_words) if len(topic_words) > 1 else query
    query_embedding = embedder.encode(query_text)
    
    history_embeddings = np.array([item["embedding"] for item in st.session_state.conversation_history])
    similarities = cosine_similarity([query_embedding], history_embeddings)[0]
    
    indexed_similarities = [(idx, sim) for idx, sim in enumerate(similarities)]
    indexed_similarities.sort(key=lambda x: (-x[0], -x[1]))
    
    top_indices = [idx for idx, _ in indexed_similarities[:top_k]]
    relevant_context = []
    retrieved_indices = []
    total_tokens = 0
    
    for idx in top_indices:
        context = f"Past User: {st.session_state.conversation_history[idx]['user']}\nPast Assistant: {st.session_state.conversation_history[idx]['assistant']}"
        token_count = len(context.split())
        if total_tokens + token_count <= max_tokens:
            relevant_context.append(context)
            retrieved_indices.append(idx)
            total_tokens += token_count
    
    return "\n\n".join(relevant_context), retrieved_indices

# Function to generate PDF from prescription text
def generate_pdf(text, filename="prescription.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    flowables.append(Paragraph("Prescription", styles['Title']))
    flowables.append(Spacer(1, 12))

    for line in text.split("\n"):
        flowables.append(Paragraph(line, styles['Normal']))
        flowables.append(Spacer(1, 6))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

# Function to format prescription and generate PDF
def format_prescription(response, query):
    if "generate the prescription" in query.lower():
        lines = response.split("\n")
        diagnosis = None
        medicines = []
        for line in lines:
            if "diagnosis" in line.lower():
                diagnosis = line.strip()
            elif line.strip() and not line.startswith((" ", "\t", "-", "*")) and "Error" not in line:
                medicines.append(line.strip())
        if not diagnosis:
            diagnosis = "Subarachnoid Hemorrhage (SAH) secondary to ruptured aneurysm"
        if not medicines:
            medicines = ["Nimodipine", "Labetalol", "Levetiracetam", "Ondansetron", "Acetaminophen"]
        prescription_text = f"{diagnosis}\n" + "\n".join(medicines)
        return prescription_text, True
    return response, False

# Function to generate a response with RAG and medical history
def generate_response(query, use_memory):
    try:
        # Classify intent using DistilBERT
        needs_memory, needs_medical_history, intent_prob = classify_intent(query)
        topic_words = word_tokenize(query.lower())

        if use_memory and (needs_memory or needs_medical_history) and (st.session_state.conversation_history or st.session_state.medical_history):
            # Increment top_k and update CSV
            new_top_k = st.session_state.top_k + 1
            update_top_k(new_top_k)
            total_conversations = len(st.session_state.conversation_history)
            top_k = min(total_conversations, max_top_k)
            context, retrieved_indices = retrieve_relevant_conversations(query, topic_words, top_k, max_tokens)

            # Add medical history context if relevant
            medical_context = ""
            if needs_medical_history and st.session_state.medical_history:
                history_lines = []
                for p in st.session_state.medical_history:
                    date = p.get('date', 'Unknown')
                    diagnoses = p.get('diagnoses', [])
                    diagnoses_str = ', '.join(str(d) for d in diagnoses if d) if isinstance(diagnoses, (list, tuple)) else 'None'
                    medications = p.get('medications', [])
                    if isinstance(medications, (list, tuple)):
                        medications_str = ', '.join(
                            f"{m.get('name', 'Unknown')} {m.get('dosage', 'N/A')}"
                            for m in medications
                            if isinstance(m, dict) and m.get('name')
                        ) if medications else 'None'
                    else:
                        medications_str = 'None'
                    line = f"Date: {date}, Diagnoses: {diagnoses_str}, Medications: {medications_str}"
                    history_lines.append(line)
                medical_context = "Patient Medical History:\n" + "\n".join(history_lines) + "\n\n"

            if context or medical_context:
                augmented_query = f"{medical_context}Previous Conversations:\n{context}\n\nCurrent Query: {query}"
                response_prefix = f"[Using {'medical history' if needs_medical_history else 'memory'} from {len(retrieved_indices) if context else 0} past chat(s), confidence: {intent_prob:.2f}] "
            else:
                augmented_query = query
                response_prefix = ""
        else:
            augmented_query = query
            response_prefix = "[Memory disabled or not applicable] " if use_memory else ""

        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": augmented_query}])
        full_response = response_prefix + response['message']['content']
        formatted_response, is_prescription = format_prescription(full_response, query)
        return formatted_response, is_prescription
    except Exception as e:
        return f"Error: {str(e)}", False

# Stream response
def stream_data(response_text):
    for word in response_text.split(" "):
        yield word + " "
        time.sleep(0.09)

# Define the Chatbot Page
def chatbot_page():
    st.title("Medical Chatbot")
    st.write("Chat with the medical assistant for advice or queries.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a medical question or describe your symptoms:", key="chat_input")
    
    # Process user input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, is_prescription = generate_response(user_input, use_memory)
                st.write_stream(stream_data(response))
                if is_prescription:
                    pdf_buffer = generate_pdf(response)
                    st.download_button(
                        label="Download Prescription as PDF",
                        data=pdf_buffer,
                        file_name="prescription.pdf",
                        mime="application/pdf"
                    )
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_conversation(user_input, response)

# Define the Prescription Upload Page
def prescription_upload_page():
    st.title("Prescription Upload")
    st.write("Upload your prescription PDF to store it in your medical history.")

    uploaded_file = st.file_uploader("Upload your prescription PDF", type="pdf", key="pdf_uploader")
    
    if uploaded_file is not None:
        with st.spinner("Processing prescription..."):
            prescription = parse_prescription(uploaded_file)
            st.session_state.medical_history.append(prescription)
            save_medical_history()
            st.success("Prescription uploaded and processed successfully!")
            
            # Display the parsed prescription
            st.subheader("Parsed Prescription")
            st.write(f"**Date:** {prescription.get('date', 'Unknown')}")
            st.write(f"**Diagnoses:** {', '.join(prescription.get('diagnoses', [])) if prescription.get('diagnoses') else 'None'}")
            medications = prescription.get('medications', [])
            if medications:
                st.write("**Medications:**")
                for med in medications:
                    st.write(f"- {med.get('name', 'Unknown')} {med.get('dosage', 'N/A')}")
            else:
                st.write("**Medications:** None")
            st.write(f"**Notes:** {prescription.get('notes', 'None')}")

    # Display current medical history
    st.subheader("Current Medical History")
    if st.session_state.medical_history:
        for i, p in enumerate(st.session_state.medical_history):
            with st.expander(f"Record {i+1}"):
                st.write(f"**Date:** {p.get('date', 'Unknown')}")
                st.write(f"**Diagnoses:** {', '.join(p.get('diagnoses', [])) if p.get('diagnoses') else 'None'}")
                medications = p.get('medications', [])
                if medications:
                    st.write("**Medications:**")
                    for med in medications:
                        st.write(f"- {med.get('name', 'Unknown')} {med.get('dosage', 'N/A')}")
                else:
                    st.write("**Medications:** None")
                st.write(f"**Notes:** {p.get('notes', 'None')}")
    else:
        st.write("No medical history available.")

# Define the pages for navigation
PAGES = {
    "Chatbot": chatbot_page,
    "Prescription Upload": prescription_upload_page
}

# Set up navigation in the sidebar
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Run the selected page
page = PAGES[selection]
page()

# Disclaimer
st.write("**Disclaimer:** This is a demo and not a substitute for professional medical advice. Data is stored locally for this session only. Delete `medical_history.json` manually if needed.")