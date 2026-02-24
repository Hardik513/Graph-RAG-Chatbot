import streamlit as st
import networkx as nx
import spacy
import matplotlib.pyplot as plt

# Initialization
st.set_page_config(page_title="Graph RAG Chatbot", layout="wide")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_transformer_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

nlp = load_spacy()
llm_tokenizer, llm_model = load_transformer_model()

# Initialize session state for multi-user memory
if "graphs" not in st.session_state:
    st.session_state.graphs = {"User_A": nx.DiGraph(), "User_B": nx.DiGraph()}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"User_A": [], "User_B": []}

# Sidebar UI
st.sidebar.title("⚙️ Settings")
st.sidebar.success("✅ Using Local Transformer (Flan-T5) & spaCy Extraction")
current_user = st.sidebar.selectbox("👤 Select User", ["User_A", "User_B"])

# Clear Memory Button
if st.sidebar.button("🗑️ Clear Memory"):
    st.session_state.graphs[current_user] = nx.DiGraph()
    st.session_state.chat_history[current_user] = []
    st.rerun()

# Core Functions

def extract_triplets_with_spacy_robust(text, current_user):
    """Ultra-robust extraction that splits sentences perfectly at the root verb."""
    doc = nlp(text)
    triplets = []
    
    for token in doc:
        if token.dep_ == "ROOT":
            relation = token.lemma_.upper()
            subject = " ".join([t.text for t in doc[:token.i]]).strip()
            obj = " ".join([t.text for t in doc[token.i + 1:]]).strip()
            
            if subject and obj:
                subject_lower = subject.lower()
                if subject_lower in ["i", "my", "me"]:
                    subject = current_user
                elif subject_lower.startswith("my "):
                    subject = current_user + "'s " + subject[3:]
                    
                if obj.lower() in ["i", "my", "me"]:
                    obj = current_user
                    
                triplets.append({"subject": subject, "relation": relation, "object": obj})
                
    return triplets

def retrieve_context(graph, query, current_user):
    """Brute-force keyword search so the chatbot never misses a fact."""
    query_words = set(query.lower().replace('?', '').split())
    context_triplets = []
    
    for u, v, data in graph.edges(data=True):
        edge_text = f"{u} {data.get('relation', '')} {v}".lower()
        
        # 1. If any major word from the query matches the graph (e.g., "live", "pizza")
        if any(word in edge_text for word in query_words if len(word) > 2):
            context_triplets.append(f"{u} {data['relation']} {v}")
            
        # 2. If the user asks about themselves ("I", "my"), pull all their personal facts
        if any(w in query_words for w in ["i", "my", "me", "am"]):
            if current_user.lower() in str(u).lower() or current_user.lower() in str(v).lower():
                context_triplets.append(f"{u} {data['relation']} {v}")
                
    return list(set(context_triplets))

def answer_query_with_llm(query, context, current_user):
    """Uses Flan-T5 to answer the question strictly based on the graph context."""
    if not context:
        return "I don't have enough information in my memory to answer that yet."
    
    # Translate "User_A" back to "The user" so the LLM understands it
    clean_context = [c.replace(current_user, "The user") for c in context]
    context_str = ". ".join(clean_context)
    
    prompt = f"""Read the context and answer the question.
Context: {context_str}
Question: {query}
Short Answer: """
    
    inputs = llm_tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=50)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main UI 
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

    # Smarter Question Detection
    clean_input = user_input.strip().lower()
    question_starters = ("who", "what", "where", "when", "why", "how", "can", "do", "does", "is", "are")
    is_question = clean_input.endswith('?') or clean_input.startswith(question_starters)
    
    with st.chat_message("assistant"):
        if is_question:
            # RAG FLOW
            context = retrieve_context(st.session_state.graphs[current_user], user_input, current_user)
            response = answer_query_with_llm(user_input, context, current_user)
            st.markdown(response)
            
            with st.expander("🔍 Retrieved Graph Context"):
                st.write(context if context else "No relevant facts found.")
                
            st.session_state.chat_history[current_user].append({"role": "assistant", "content": response})
            
        else:
            # MEMORY UPDATE FLOW
            with st.spinner("Updating knowledge graph..."):
                triplets = extract_triplets_with_spacy_robust(user_input, current_user)
                
                if triplets:
                    graph = st.session_state.graphs[current_user]
                    for t in triplets:
                        graph.add_edge(t['subject'], t['object'], relation=t['relation'])
                    
                    response = f"Got it! I've updated your memory with {len(triplets)} new fact(s). Check the live graph above."
                else:
                    response = "I couldn't extract any clear facts from that statement."
            
            st.markdown(response)
            st.session_state.chat_history[current_user].append({"role": "assistant", "content": response})
            st.rerun()
