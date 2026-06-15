import streamlit as st
import os
import pandas as pd
from summarizer import (
    DocumentSummarizer,
    read_file,
    generate_txt_bytes,
    generate_pdf_bytes,
    generate_combined_txt_bytes,
    generate_pdf_bytes_multi,
)

# --- Page Config ---
st.set_page_config(
    page_title="Summarizer AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Robust CSS Injection ---
st.markdown("""
<style>
    /* Hide specific elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Modern Headers */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.05rem !important;
        background: linear-gradient(to right, #F8FAFC, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    /* Metric Boxes */
    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 1.5rem;
        border-radius: 16px;
    }

    /* Results Containers */
    .stExpander {
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        background-color: #1E293B !important;
    }

    /* Fix for text rendering as plain text */
    div.stMarkdown p {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def get_model():
    return DocumentSummarizer()

model = get_model()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ✨ Summarizer AI")
    st.markdown("---")
    choice = st.selectbox("NAVIGATION", ["Home", "Analytics", "About"])
    
    st.markdown("---")
    st.markdown("#### Settings")
    method = st.radio("Method", ["Simple Frequency", "TF-IDF"])
    num_sent = st.slider("Sentences", 1, 10, 3)
    
    st.markdown("---")
    st.caption("Premium NLP Engine v2.3")

# --- Home Page ---
if choice == "Home":
    st.markdown("<h1>Summarizer AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>The modern way to digest long-form content.</p>", unsafe_allow_html=True)

    # Input Section
    tab1, tab2, tab3 = st.tabs(["📝 Text", "📄 Single File", "📚 Multiple"])
    
    documents_input = []
    
    with tab1:
        text_data = st.text_area("Content", height=300, placeholder="Paste text here...", label_visibility="collapsed")
        if text_data.strip():
            documents_input = [("Pasted Text", text_data)]
            
    with tab2:
        file = st.file_uploader("Upload File", type=["txt", "pdf"], label_visibility="collapsed")
        if file:
            temp_path = f"temp_{file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                text_data = read_file(temp_path)
                documents_input = [(file.name, text_data)]
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    with tab3:
        files = st.file_uploader("Upload Files", type=["txt", "pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if files:
            for f in files:
                temp_path = f"temp_{f.name}"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(f.getbuffer())
                    text_data = read_file(temp_path)
                    documents_input.append((f.name, text_data))
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Summary", type="primary", use_container_width=True):
        if not documents_input:
            st.warning("Please provide input.")
        else:
            with st.spinner("Processing..."):
                results = []
                for name, text in documents_input:
                    try:
                        res = model.summarize(text, method, num_sent)
                        if res.strip():
                            results.append({"name": name, "original": text, "summary": res})
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                if results:
                    st.session_state['documents'] = results
                    st.session_state['method'] = method

    # Results
    if 'documents' in st.session_state and st.session_state['documents']:
        st.markdown("---")
        st.subheader("Generated Summaries")
        
        for doc in st.session_state['documents']:
            with st.expander(f"📄 {doc['name']}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original**")
                    st.info(doc['original'][:800] + "...")
                with c2:
                    st.markdown("**Summary**")
                    st.success(doc['summary'])
                
                # Actions
                a1, a2 = st.columns(2)
                a1.download_button("⬇️ TXT", generate_txt_bytes(doc['summary']), f"{doc['name']}.txt", key=f"t_{doc['name']}")
                try:
                    a2.download_button("⬇️ PDF", generate_pdf_bytes(doc['summary']), f"{doc['name']}.pdf", key=f"p_{doc['name']}")
                except:
                    pass

elif choice == "Analytics":
    st.markdown("<h1>Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Document insights and metrics.</p>", unsafe_allow_html=True)
    
    docs = st.session_state.get('documents', [])
    if not docs:
        st.warning("Summarize a document first.")
    else:
        selected = st.selectbox("Document", [d['name'] for d in docs])
        doc = next(d for d in docs if d['name'] == selected)
        
        stats = model.get_analytics(doc['original'], st.session_state.get('method', 'Simple Frequency'))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Sentences", stats['num_sentences'])
        m2.metric("Words", stats['num_words'])
        m3.metric("Keywords", stats['keywords'][0] if stats['keywords'] else "-")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame(list(stats['word_freq'].items()), columns=['Word', 'Count']).set_index('Word'))

else:
    st.markdown("<h1>About</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Technical specifications.</p>", unsafe_allow_html=True)
    st.markdown("""
    **Summarizer AI** is a high-performance NLP tool designed for speed and clarity.
    
    - **Methodology:** Extractive Summarization (Frequency & TF-IDF)
    - **Stack:** Python, Streamlit, NLTK, Scikit-learn
    - **Design:** Modern Dark Slate Theme
    """)
    st.info("Developed by AI Intern Team")
