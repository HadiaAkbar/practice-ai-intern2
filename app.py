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

# Page configuration
st.set_page_config(
    page_title="SUMMARIZER.AI",
    page_icon="🕊️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Elegant Modern Pastel CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');
    
    :root {
        --bg-color: #FAFAFA;
        --text-color: #4A4A4A;
        --text-light: #7D7D7D;
        --pastel-blue: #EBF1FF;
        --pastel-purple: #F3EBFF;
        --pastel-pink: #FFEBF1;
        --pastel-green: #EBFFFA;
        --white: #FFFFFF;
        --border-soft: #F0F0F0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
        background-color: var(--bg-color);
    }

    /* Elegant Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: #2C2C2C;
    }

    /* Airy Header */
    .header-container {
        padding: 6rem 2rem 4rem 2rem;
        text-align: center;
        background-color: var(--bg-color);
    }

    .header-container h1 {
        font-size: 4.5rem;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }

    .tagline {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.1rem;
        color: var(--text-light);
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: var(--white) !important;
        border-right: 1px solid var(--border-soft);
    }

    .sidebar-logo {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2C2C2C;
        margin-bottom: 3rem;
    }

    /* Elegant Pastel Cards */
    .elegant-card {
        background: var(--white);
        padding: 3rem;
        border-radius: 40px;
        border: 1px solid var(--border-soft);
        box-shadow: 0 15px 40px rgba(0,0,0,0.02);
        margin-bottom: 2rem;
    }

    /* Soft 3D Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 100px;
        padding: 1rem 2rem;
        background-color: #2C2C2C;
        color: var(--white);
        font-weight: 500;
        border: none;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }

    .stButton>button:hover {
        background-color: #000000;
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        color: var(--white);
    }

    /* Minimalist Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 40px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-soft);
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 15px 0px;
        background-color: transparent;
        color: var(--text-light);
        font-weight: 400;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        color: #2C2C2C !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2C2C2C !important;
    }

    /* Result Panels */
    .result-box {
        padding: 2rem;
        border-radius: 30px;
        margin-bottom: 1.5rem;
        line-height: 1.8;
    }

    .source-panel { background-color: var(--pastel-blue); border: 1px solid #DCE6FF; }
    .summary-panel { background-color: var(--pastel-purple); border: 1px solid #EBDCFF; }

    /* Elegant Metrics */
    .metric-item {
        text-align: center;
        padding: 2rem;
        background-color: var(--white);
        border-radius: 30px;
        border: 1px solid var(--border-soft);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: #2C2C2C;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return DocumentSummarizer()

model = get_model()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Summarizer.</div>', unsafe_allow_html=True)
    choice = st.selectbox("WORKSPACE", ["Home", "Analytics", "About"])
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.7rem; letter-spacing: 1px; color: var(--text-light); text-transform: uppercase;'>Settings</p>", unsafe_allow_html=True)
    method = st.radio("Method", ["Simple Frequency", "TF-IDF"], label_visibility="collapsed")
    num_sent = st.slider("Depth", 1, 10, 3)
    st.markdown("---")
    st.caption("AI Intern • Fifth Edition")

# Main Page
if choice == "Home":
    st.markdown('<div class="header-container"><h1>Knowledge, Refined.</h1><p class="tagline">An elegant AI for the modern reader</p></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="elegant-card">', unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["Manual Entry", "File Upload", "Batch Mode"])
        documents_input = []
        
        with t1:
            st.markdown("<br>", unsafe_allow_html=True)
            text_data = st.text_area("INPUT", height=300, placeholder="Paste your text here...", label_visibility="collapsed")
            if text_data.strip():
                documents_input = [("Manual Entry", text_data)]
        
        with t2:
            st.markdown("<br>", unsafe_allow_html=True)
            file = st.file_uploader("UPLOAD", type=["txt", "pdf"], label_visibility="collapsed")
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
        
        with t3:
            st.markdown("<br>", unsafe_allow_html=True)
            files = st.file_uploader("BATCH", type=["txt", "pdf"], accept_multiple_files=True, label_visibility="collapsed")
            if files:
                for f in files:
                    temp_path = f"temp_{f.name}"
                    try:
                        with open(temp_path, "wb") as f_out:
                            f_out.write(f.getbuffer())
                        text_data = read_file(temp_path)
                        documents_input.append((f.name, text_data))
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_btn, _ = st.columns([1, 2])
    with c_btn:
        if st.button("Generate Summary"):
            if not documents_input:
                st.warning("Please provide some input.")
            else:
                with st.spinner("Analyzing..."):
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

    if 'documents' in st.session_state and st.session_state['documents']:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3>Analysis Report</h3>", unsafe_allow_html=True)
        for doc in st.session_state['documents']:
            with st.expander(f"Report: {doc['name']}", expanded=True):
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.markdown("<p style='font-size: 0.7rem; letter-spacing: 1px; color: var(--text-light); text-transform: uppercase; margin-bottom: 1rem;'>Original Content</p>", unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box source-panel">{doc["original"][:1500]}...</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown("<p style='font-size: 0.7rem; letter-spacing: 1px; color: var(--text-light); text-transform: uppercase; margin-bottom: 1rem;'>AI Core Summary</p>", unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box summary-panel" style="font-weight: 500;">{doc["summary"]}</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                a1, a2, _ = st.columns([1, 1, 2])
                a1.download_button("Export TXT", generate_txt_bytes(doc['summary']), f"{doc['name']}_summary.txt", key=f"t_{doc['name']}")
                try:
                    a2.download_button("Export PDF", generate_pdf_bytes(doc['summary']), f"{doc['name']}_summary.pdf", key=f"p_{doc['name']}")
                except:
                    pass

elif choice == "Analytics":
    st.markdown('<div class="header-container"><h1>Data Insights.</h1><p class="tagline">The science of summarization</p></div>', unsafe_allow_html=True)
    docs = st.session_state.get('documents', [])
    if not docs:
        st.info("Run a summary on the Home page to see insights.")
    else:
        selected = st.selectbox("Select Document", [d['name'] for d in docs])
        doc = next(d for d in docs if d['name'] == selected)
        stats = model.get_analytics(doc['original'], st.session_state.get('method', 'Simple Frequency'))
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-item"><p class="metric-label">Sentences</p><p class="metric-value">{stats["num_sentences"]}</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-item"><p class="metric-label">Words</p><p class="metric-value">{stats["num_words"]}</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-item"><p class="metric-label">Primary Keyword</p><p class="metric-value" style="font-size: 1.5rem; padding-top: 10px;">{stats["keywords"][0].upper() if stats["keywords"] else "N/A"}</p></div>', unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3>Frequency Analysis</h3>", unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame(list(stats['word_freq'].items()), columns=['Word', 'Count']).set_index('Word'))

else:
    st.markdown('<div class="header-container"><h1>System.</h1><p class="tagline">Under the hood</p></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="elegant-card">
        <h3>Architecture</h3>
        <p style="color: var(--text-light); line-height: 1.8;">
            This platform uses a custom implementation of extractive summarization algorithms, leveraging statistical term weighting and frequency distribution 
            to identify the core essence of any given document.
        </p>
        <br>
        <p style="font-size: 0.8rem; color: var(--text-light);">Build 5.0 • Modern Elegant Edition</p>
    </div>
    """, unsafe_allow_html=True)
