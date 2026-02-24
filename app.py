import streamlit as st
import networkx as nx
import spacy
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Initialization ---
st.set_page_config(page_title="Graph RAG Chatbot", layout="wide")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_transformer_model():
    # Bypassing the pipeline wrapper entirely to prevent task inference errors
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

nlp = load_spacy()
llm_model = load_transformer_model()

# Initialize session state for multi-user memory
if "graphs" not in st.session_state:
    st.session_state.graphs = {"User_A": nx.DiGraph(), "User_B": nx.DiGraph()}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"User_A": [], "User_B": []}

# --- Sidebar UI ---
st.sidebar.title("⚙️ Settings")
st.sidebar.success("✅ Using Local Transformer (Flan-T5) & spaCy Extraction")
current_user = st.sidebar.selectbox("👤 Select User (Multi-User Memory)", ["User_A", "User_B"])

# --- Core Functions ---

def extract_triplets_with_spacy_robust(text, current_user):
    """Uses spaCy to reliably extract SVO triplets, including prepositional phrases."""
    doc = nlp(text)
    triplets = []
    
    for token in doc:
        # Find the main verb of the sentence
        if token.dep_ == "ROOT": 
            relation = token.lemma_.upper()
            
            # Find the subject (looking left of the verb)
            subject = None
            for child in token.lefts:
                if child.dep_ in ("nsubj", "nsubjpass", "expl"):
                    subject = " ".join([t.text for t in child.subtree])
            
            # Find the object (looking right of the verb)
            obj = None
            for child in token.rights:
                # Direct objects, attributes, etc.
                if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                    obj = " ".join([t.text for t in child.subtree])
                # NEW FIX: Handle prepositions like "in Goa", "at Microsoft", "with Emma"
                elif child.dep_ == "prep":
                    obj = " ".join([t.text for t in child.subtree])
            
            # Fallback for "is/are" sentences
            if not obj and relation in ["BE", "IS", "ARE"]:
                rights = list(token.rights)
                if rights:
                    obj = " ".join([t.text for t in rights[0].subtree])
                    
            if subject and obj:
                # Advanced Pronoun Normalization
                subject_lower = subject.lower()
                if subject_lower in ["i", "my", "me"]:
                    subject = current_user
                elif subject_lower.startswith("my "):
                    subject = current_user + "'s " + subject[3:]
                    
                if obj.lower() in ["i", "my", "me"]:
                    obj = current_user
                    
                triplets.append({"subject": subject, "relation": relation, "object": obj})
                
    return triplets

def extract_entities_with_spacy(query):
    """Uses spaCy to extract key entities/nouns from a query for graph retrieval."""
    doc = nlp(query)
    entities = [ent.text.lower() for ent in doc.ents]
    nouns = [chunk.text.lower() for chunk in doc.noun_chunks]
    # Combine and deduplicate
    return list(set(entities + nouns + ["user", "i", "my", "me"]))

def retrieve_context(graph, query):
    """Retrieves 1-hop neighborhood from the graph based on spaCy entities."""
    query_entities = extract_entities_with_spacy(query)
    context_triplets = []
    
    for u, v, data in graph.edges(data=True):
        # If the subject or object overlaps with our query entities, pull the fact
        if any(ent in u.lower() or ent in v.lower() or u.lower() in ent or v.lower() in ent for ent in query_entities):
            context_triplets.append(f"{u} {data['relation']} {v}")
            
    return list(set(context_triplets))

def answer_query_with_llm(query, context, current_user):
    """Uses Flan-T5 to answer the question strictly based on the graph context."""
    if not context:
        return "I don't have enough information in my memory to answer that yet."
    
    context_str = ". ".join(context)
    prompt = f"""
    Context: {context_str}. (Note: '{current_user}' refers to 'I' or 'me').
    Question: {query}
    Answer based purely on the context provided:
    """
    
    tokenizer, model = llm_model
    
    # Tokenize and generate manually to bypass pipeline errors
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Main UI ---
st.title("🧠 Graph RAG Chatbot")
st.write(f"Currently chatting as: **{current_user}**")

# Display Live Graph Visualization
with st.expander("👁️ View Live Knowledge Graph for " + current_user):
    graph = st.session_state.graphs[current_user]
    if len(graph.nodes) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
                node_size=2000, font_size=9, font_weight='bold', ax=ax)
        edge_labels = nx.get_edge_attributes(graph, 'relation')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        st.pyplot(fig)
    else:
        st.write("Graph is currently empty. Add some facts!")

# Render Chat History
for msg in st.session_state.chat_history[current_user]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Area
user_input = st.chat_input("Tell me a fact, or ask me a question (ending with ?)")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history[current_user].append({"role": "user", "content": user_input})

    # Determine intent: Question vs. Statement
    # Determine intent: Question vs. Statement
    clean_input = user_input.strip().lower()
    question_starters = ("who", "what", "where", "when", "why", "how", "can", "do", "does", "is", "are")
    is_question = clean_input.endswith('?') or clean_input.startswith(question_starters)
    
    with st.chat_message("assistant"):
        if is_question:
            # RAG FLOW
            context = retrieve_context(st.session_state.graphs[current_user], user_input)
            response = answer_query_with_llm(user_input, context, current_user)
            st.markdown(response)
            
            # Show debug context
            with st.expander("🔍 Retrieved Graph Context"):
                st.write(context if context else "No relevant facts found.")
                
            # Append question answer to history
            st.session_state.chat_history[current_user].append({"role": "assistant", "content": response})
            
        else:
            # MEMORY UPDATE FLOW
            with st.spinner("Updating knowledge graph..."):
                triplets = extract_triplets_with_spacy_robust(user_input, current_user)
                
                if triplets:
                    graph = st.session_state.graphs[current_user]
                    for t in triplets:
                        # Add to graph directly from the spaCy output
                        graph.add_edge(t['subject'], t['object'], relation=t['relation'])
                    
                    response = f"Got it! I've updated your memory with {len(triplets)} new fact(s). Check the live graph above."
                else:
                    response = "I couldn't extract any clear facts from that statement. Try using a simple Subject-Verb-Object sentence (e.g., 'Alex loves pizza')."
            
            st.markdown(response)
            
            # Save history BEFORE rerunning
            st.session_state.chat_history[current_user].append({"role": "assistant", "content": response})
            
            # Rerun to update the live graph visual
            st.rerun()