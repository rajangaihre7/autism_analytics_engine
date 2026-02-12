import streamlit as st
import pandas as pd
import os

# --- IMPORT YOUR NEW MODULES ---
from modules import executive, efficacy, drivers, perspective, nlp_view, drilldown

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Powered Storytelling Platform Research Analytics", page_icon="🧩", layout="wide")

# --- CSS ---
st.markdown("""<style>.metric-card {background-color: #f8f9fa; border-left: 5px solid #2e86c1;}</style>""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    paths = {
        'clean': os.path.join(project_root, 'data', 'silver', 'After_transformation_Data', 'silver_cleaned.csv'),
        'stats': os.path.join(project_root, 'data', 'gold', 'statistical_results', 'gold_statistical_answers.csv'),
        'nlp': os.path.join(project_root, 'data', 'gold', 'nlp_results', 'gold_nlp_full_session_sentiment.csv'),
        'keywords': os.path.join(project_root, 'data', 'gold', 'nlp_results', 'gold_nlp_keyword_trends.csv')
    }
    
    data = {}
    if os.path.exists(paths['clean']): data['df'] = pd.read_csv(paths['clean'])
    else: return None
        
    if os.path.exists(paths['stats']): data['stats'] = pd.read_csv(paths['stats'])
    if os.path.exists(paths['nlp']): data['nlp'] = pd.read_csv(paths['nlp'])
    if os.path.exists(paths['keywords']): data['keywords'] = pd.read_csv(paths['keywords'])
    
    return data

data_dict = load_data()
if not data_dict: st.stop()

# Unpack Data
df = data_dict['df']
stats_df = data_dict.get('stats', pd.DataFrame())
nlp_df = data_dict.get('nlp', pd.DataFrame())
kw_df = data_dict.get('keywords', pd.DataFrame())

# --- NAVIGATION ---
st.sidebar.image("https://img.icons8.com/color/96/000000/autism.png", width=70)
st.sidebar.title("Intervention Analytics")
page = st.sidebar.radio("Modules:", [
    "1. Executive Overview",
    "2. Efficacy & Safety",
    "3. Drivers & Mechanisms",
    "4. Perspective Analysis",
    "5. Qualitative NLP",
    "6. Participant Drill-Down"
])

# --- ROUTING LOGIC ---
if page == "1. Executive Overview":
    executive.show(df)

elif page == "2. Efficacy & Safety":
    efficacy.show(df, stats_df)

elif page == "3. Drivers & Mechanisms":
    drivers.show(df, stats_df)

elif page == "4. Perspective Analysis":
    perspective.show(df)

elif page == "5. Qualitative NLP":
    nlp_view.show(nlp_df, kw_df)

elif page == "6. Participant Drill-Down":
    drilldown.show(df)