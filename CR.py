import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd

# Setup
st.set_page_config(page_title="Clause Risk Evaluator", layout="wide")
st.title("📜 Clause Classifier & Risk Evaluator")

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

model = load_model()
nlp = load_spacy()

# State to hold reference clauses
if "clause_library" not in st.session_state:
    st.session_state.clause_library = []

# Function: Embed and store clause
def add_clause(clause_type, clause_text):
    embedding = model.encode(clause_text)
    st.session_state.clause_library.append({
        "clause_type": clause_type,
        "clause_text": clause_text,
        "embedding": embedding
    })

# Function: Extract key NLU concepts
def extract_key_concepts(text):
    doc = nlp(text)
    verbs = [token.lemma_ for token in doc if token.pos_ in ("VERB", "AUX")]
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    return set(verbs + noun_chunks)

# Function: Compare NLU content
def check_missing_concepts_nlu(reference, new_clause):
    ref = extract_key_concepts(reference)
    new = extract_key_concepts(new_clause)
    return list(ref - new)

# Function: Risk assessment
def assess_risk(similarity, missing_concepts):
    if similarity < 0.75 or len(missing_concepts) > 2:
        return "🔴 HIGH", "YES"
    elif similarity < 0.85 or len(missing_concepts) > 0:
        return "🟠 MEDIUM", "POSSIBLE"
    else:
        return "🟢 LOW", "NO"

# Function: Evaluate new clause
def evaluate_clause(new_clause_text):
    if not st.session_state.clause_library:
        return None, None, None, None, None

    new_vec = model.encode(new_clause_text)

    scored = []
    for clause in st.session_state.clause_library:
        score = cosine_similarity([new_vec], [clause["embedding"]])[0][0]
        scored.append({
            "clause_type": clause["clause_type"],
            "reference_text": clause["clause_text"],
            "similarity": score
        })

    top_match = sorted(scored, key=lambda x: x["similarity"], reverse=True)[0]
    missing = check_missing_concepts_nlu(top_match["reference_text"], new_clause_text)
    risk, deviation = assess_risk(top_match["similarity"], missing)

    return top_match, risk, deviation, missing, top_match["similarity"]

# --- Streamlit Tabs ---
tab1, tab2 = st.tabs(["📘 Clause Trainer", "🧪 New Clause Evaluator"])

# --- Tab 1: Trainer ---
with tab1:
    st.subheader("Add Reference Clauses")
    with st.form("clause_form"):
        clause_type = st.text_input("Clause Type (e.g., Confidentiality)")
        clause_text = st.text_area("Clause Text", height=200)
        submitted = st.form_submit_button("➕ Add to Library")
        if submitted and clause_type and clause_text:
            add_clause(clause_type, clause_text)
            st.success("Clause added successfully!")

    if st.session_state.clause_library:
        st.markdown("### 📚 Current Clause Library")
        df = pd.DataFrame([{
            "Clause Type": c["clause_type"],
            "Clause Text": c["clause_text"]
        } for c in st.session_state.clause_library])
        st.dataframe(df, use_container_width=True)

# --- Tab 2: Evaluator ---
with tab2:
    st.subheader("Evaluate New Clause")
    new_clause_input = st.text_area("Paste your new clause below:", height=150)

    if st.button("🔍 Analyze Clause"):
        if not new_clause_input.strip():
            st.warning("Please enter a clause.")
        elif not st.session_state.clause_library:
            st.warning("Please add reference clauses first in Tab 1.")
        else:
            top_match, risk, deviation, missing, similarity = evaluate_clause(new_clause_input)

            if top_match:
                st.markdown("### 🧠 Evaluation Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Clause Type", top_match["clause_type"])
                    st.metric("Similarity Score", f"{similarity:.3f}")
                    st.metric("Deviation", deviation)
                    st.metric("Risk Level", risk)

                with col2:
                    st.markdown("**Missing Concepts (NLU):**")
                    st.write(missing if missing else "✅ None")

                st.markdown("### 📄 Closest Reference Clause")
                st.code(top_match["reference_text"])

                st.markdown("### ✍️ Your Clause")
                st.code(new_clause_input.strip())
