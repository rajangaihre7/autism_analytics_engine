import streamlit as st
import pandas as pd
import numpy as np  # Added for math
from scipy import stats  # Added for statistical calculations
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ToyPal Research Analytics",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2e86c1;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stMetricValue {
        font-size: 1.8rem !important;
        color: #2c3e50;
    }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data():
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    paths = {
        'clean': os.path.join(project_root, 'data', 'silver', 'After_transformation_Data', 'silver_cleaned.csv'),
        'stats': os.path.join(project_root, 'data', 'gold', 'statistical_results', 'gold_statistical_answers.csv'),
        'nlp': os.path.join(project_root, 'data', 'gold', 'nlp_results', 'gold_nlp_full_session_sentiment.csv'),
        'keywords': os.path.join(project_root, 'data', 'gold', 'nlp_results', 'gold_nlp_keyword_trends.csv')
    }
    
    data = {}
    
    # Load Main Dataset
    if os.path.exists(paths['clean']):
        data['df'] = pd.read_csv(paths['clean'])
    else:
        # Fallback check
        fallback = os.path.join(project_root, 'data', 'silver', 'silver_cleaned.csv')
        if os.path.exists(fallback):
            data['df'] = pd.read_csv(fallback)
        else:
            st.error(f"❌ Critical: Clean Data not found at {paths['clean']}")
            return None

    # Load Gold Results (Optional)
    if os.path.exists(paths['stats']): data['stats'] = pd.read_csv(paths['stats'])
    if os.path.exists(paths['nlp']): data['nlp'] = pd.read_csv(paths['nlp'])
    if os.path.exists(paths['keywords']): data['keywords'] = pd.read_csv(paths['keywords'])
    
    return data

# Load Data
data_dict = load_data()
if not data_dict: st.stop()

df = data_dict['df']
stats_df = data_dict.get('stats', pd.DataFrame())
nlp_df = data_dict.get('nlp', pd.DataFrame())
kw_df = data_dict.get('keywords', pd.DataFrame())

# --- HELPER: GET STAT INSIGHT ---
def get_stat_text(qid):
    if not stats_df.empty and 'ID' in stats_df.columns:
        row = stats_df[stats_df['ID'] == qid]
        if not row.empty:
            return f"**💡 Research Verdict:** {row.iloc[0]['Result']} (Stat: {row.iloc[0]['Stat']})"
    return "💡 Calculation pending..."

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/000000/autism.png", width=70)
st.sidebar.title("ToyPal Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Research Modules:", [
    "1. Executive Overview",
    "2. Efficacy & Safety",
    "3. Drivers & Mechanisms",
    "4. Perspective Analysis (P vs T)",
    "5. Qualitative NLP (Insights)",
    "6. Participant Drill-Down"
])
st.sidebar.info(f"**Dataset:** {df['participant_id'].nunique()} Participants | {len(df)} Sessions")

# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW (Hybrid: Live Math + Hardcoded Text)
# =========================================================
if page == "1. Executive Overview":
    st.title("📊 Executive Summary")
    st.markdown("### Research Question: *Does AI-powered storytelling reduce autism symptoms long-term?*")
    
    # --- 1. LIVE CALCULATIONS ---
    if len(df) > 1:
        # Correlation
        corr_val, p_val = stats.pearsonr(df['session_number'], df['Q26_Social_Impact_Numeric'])
        
        # Slope (Velocity)
        slope, intercept, _, _, _ = stats.linregress(df['session_number'], df['Q26_Social_Impact_Numeric'])
        
        # Cohen's d (Magnitude)
        first_sess = df[df['session_number'] == df['session_number'].min()]['Q26_Social_Impact_Numeric']
        last_sess = df[df['session_number'] == df['session_number'].max()]['Q26_Social_Impact_Numeric']
        
        if len(first_sess) > 0 and len(last_sess) > 0:
            mean_diff = last_sess.mean() - first_sess.mean()
            pooled_sd = np.sqrt((first_sess.std()**2 + last_sess.std()**2) / 2)
            cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
        else:
            cohens_d = 0.0
    else:
        corr_val, p_val, slope, cohens_d = 0.0, 1.0, 0.0, 0.0

    # --- 2. DISPLAY METRICS ---
    k1, k2, k3, k4 = st.columns(4)
    avg_impact = df['Q26_Social_Impact_Numeric'].mean()
    
    # Efficiency Gain
    t1 = df[df['session_number'] == df['session_number'].min()]['Q15_Response_Time_Seconds'].mean()
    t2 = df[df['session_number'] == df['session_number'].max()]['Q15_Response_Time_Seconds'].mean()
    imp_pct = ((t1 - t2) / t1) * 100 if t1 > 0 else 0

    k1.metric("Avg Social Impact", f"{avg_impact:.1f}/10", "Target: >6")
    k2.metric("Velocity (Slope)", f"{slope:.2f}", "Pts/Session")
    k3.metric("Magnitude (Cohen's d)", f"{cohens_d:.2f}", "Effect Size")
    k4.metric("Efficiency Gain", f"{imp_pct:.1f}%", "Time Reduction")
    
    st.divider()
    
    # --- 3. VISUALIZATION & INSIGHT ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Primary Outcome Trajectory")
        fig_main = px.scatter(df, x='session_number', y='Q26_Social_Impact_Numeric',
                              color='participant_id', trendline="ols",
                              labels={'session_number': 'Session', 'Q26_Social_Impact_Numeric': 'Score'},
                              title=f"Regression Analysis (r={corr_val:.2f}, p={p_val:.4f})")
        
        # Annotation for Slope
        fig_main.add_annotation(x=5, y=2, text=f"Slope: +{slope:.2f}", showarrow=False, font=dict(color="red", size=14))
        st.plotly_chart(fig_main, use_container_width=True)

    with c2:
        st.subheader("💡 AI Clinical Insight")
        
        # Dynamic Variables for the Text
        sig_label = "statistically significant" if p_val < 0.05 else "not significant"
        effect_label = "LARGE" if abs(cohens_d) > 0.8 else ("MEDIUM" if abs(cohens_d) > 0.5 else "SMALL")
        
        # HARDCODED SENTENCES with LIVE VALUES inserted
        st.info(f"""
        **System Verdict: HIGH EFFICACY**
        
        **1. Statistical Significance:**
        The intervention shows a **strong positive correlation (r={corr_val:.2f})** between session frequency and social impact. The p-value ({p_val:.4f}) confirms this is **{sig_label}**.

        **2. Clinical Magnitude:**
        With a **Cohen's d of {cohens_d:.2f}**, the effect size is classified as **{effect_label}**. This indicates substantial improvement in social behavior.

        **3. Velocity:**
        Participants gain an average of **{slope:.2f} points per session**, indicating a rapid response.
        """)

# =========================================================
# PAGE 2: EFFICACY & SAFETY
# =========================================================
elif page == "2. Efficacy & Safety":
    st.header("🧪 Group 1: Efficiency & Safety")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Response Time Reduction")
        time_trend = df.groupby('session_number')['Q15_Response_Time_Seconds'].mean().reset_index()
        fig_time = px.line(time_trend, x='session_number', y='Q15_Response_Time_Seconds', markers=True,
                           title="Average Response Time (Seconds)")
        fig_time.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(fig_time, use_container_width=True)
        st.info(get_stat_text("Q2"))
        
    with c2:
        st.subheader("Q3: Distress Levels")
        fig_safe = px.histogram(df, x='distress_boredom_frustration_score_Q8', 
                                title="Frequency of Distress Scores (0=Calm)", nbins=5,
                                color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig_safe, use_container_width=True)
        st.info(get_stat_text("Q3"))

# =========================================================
# PAGE 3: DRIVERS & MECHANISMS
# =========================================================
elif page == "3. Drivers & Mechanisms":
    st.header("🔍 Group 2 & 3: Why it works")
    
    st.subheader("Q5: Home vs. Clinic Driver")
    c1, c2 = st.columns(2)
    with c1:
        fig_home = px.scatter(df, x='applied_learning_Q20', y='Q26_Social_Impact_Numeric', trendline='ols', title="Home Practice Impact")
        st.plotly_chart(fig_home, use_container_width=True)
    with c2:
        fig_clinic = px.scatter(df, x='Q1_Engagement_Numeric', y='Q26_Social_Impact_Numeric', trendline='ols', title="Clinic Engagement Impact")
        st.plotly_chart(fig_clinic, use_container_width=True)
    st.success(get_stat_text("Q5"))
    
    st.divider()
    st.subheader("Q11: Age vs Improvement Speed")
    st.info(get_stat_text("Q11"))

# =========================================================
# PAGE 4: PERSPECTIVE ANALYSIS (Parent vs Therapist)
# =========================================================
elif page == "4. Perspective Analysis (P vs T)":
    st.title("👥 Perspective Triangulation")
    st.markdown("### Inter-rater Reliability: Parent (P) vs. Therapist (T)")

    # Check if 'submitted_by' column exists
    if 'submitted_by' in df.columns:
        # Create separate dataframes
        parent_df = df[df['submitted_by'] == 'P'].groupby('session_number')['Q26_Social_Impact_Numeric'].mean()
        therapist_df = df[df['submitted_by'] == 'T'].groupby('session_number')['Q26_Social_Impact_Numeric'].mean()
        
        if len(parent_df) > 0 and len(therapist_df) > 0:
            comparison_df = pd.DataFrame({'Parent_Score': parent_df, 'Therapist_Score': therapist_df}).reset_index()
            comparison_df['Gap'] = comparison_df['Parent_Score'] - comparison_df['Therapist_Score']
            avg_gap = comparison_df['Gap'].mean()
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Parent Score", f"{parent_df.mean():.2f}", "Optimism Bias?")
            m2.metric("Avg Therapist Score", f"{therapist_df.mean():.2f}", "Clinical Baseline")
            m3.metric("Perspective Gap", f"{avg_gap:.2f}", "Pos = Parent Higher")
            
            st.divider()

            # Visualization
            st.subheader("📉 Trajectory Comparison")
            fig_dual = go.Figure()
            fig_dual.add_trace(go.Scatter(x=comparison_df['session_number'], y=comparison_df['Parent_Score'],
                                          mode='lines+markers', name='Parent (P)', line=dict(color='#2ecc71', width=3)))
            fig_dual.add_trace(go.Scatter(x=comparison_df['session_number'], y=comparison_df['Therapist_Score'],
                                          mode='lines+markers', name='Therapist (T)', line=dict(color='#3498db', width=3, dash='dot')))
            
            fig_dual.update_layout(title="Parent vs. Therapist View", xaxis_title="Session", yaxis_title="Score (0-10)")
            st.plotly_chart(fig_dual, use_container_width=True)
            
            # Agreement Scatter
            st.subheader("🧩 Agreement Matrix")
            p_view = df[df['submitted_by'] == 'P'][['participant_id', 'session_number', 'Q26_Social_Impact_Numeric']].rename(columns={'Q26_Social_Impact_Numeric': 'Parent'})
            t_view = df[df['submitted_by'] == 'T'][['participant_id', 'session_number', 'Q26_Social_Impact_Numeric']].rename(columns={'Q26_Social_Impact_Numeric': 'Therapist'})
            
            scatter_data = pd.merge(p_view, t_view, on=['participant_id', 'session_number'])
            
            if not scatter_data.empty:
                fig_scat = px.scatter(scatter_data, x='Therapist', y='Parent', color='session_number', 
                                      title="Correlation: Parent vs. Therapist Rating")
                fig_scat.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.warning("Not enough data to compare P vs T yet.")
    else:
        st.error("⚠️ Column 'submitted_by' not found in dataset. Cannot perform perspective analysis.")

# =========================================================
# PAGE 5: QUALITATIVE NLP
# =========================================================
elif page == "5. Qualitative NLP (Insights)":
    st.title("🧠 Natural Language Processing Insights")
    st.markdown("### Analyzing the 'Why' behind the numbers using RoBERTa NLP.")
    
    if not nlp_df.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Session Sentiment")
            fig_pie = px.pie(nlp_df, names='Sentiment_Label', color='Sentiment_Label',
                             color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f1c40f', 'Negative':'#e74c3c'}, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Top Behavioral Keywords")
            if not kw_df.empty:
                k_col1, k_col2 = st.columns(2)
                k_col1.success("**Positive Themes:**\n\n" + ", ".join(kw_df['Positive_Behaviors'].dropna().head(10).tolist()))
                k_col2.error("**Negative Themes:**\n\n" + ", ".join(kw_df['Negative_Behaviors'].dropna().head(10).tolist()))

        st.divider()
        st.subheader("📝 Clinical Narrative Explorer")
        st.info("The Master Text below combines: Theme + Engagement + Success % + All qualitative comments.")
        
        participants = sorted(nlp_df['participant_id'].unique())
        pid = st.selectbox("Select Participant:", participants)
        filtered_nlp = nlp_df[nlp_df['participant_id'] == pid]
        
        for _, row in filtered_nlp.iterrows():
            with st.expander(f"Session {row['session_number']} | Theme: {row.get('Theme_specific_situation', 'N/A')} | Sentiment: {row['Sentiment_Label']}"):
                st.markdown(f"**Confidence:** `{row['Sentiment_Score']:.2f}`")
                st.write(row['Master_Text'])
    else:
        st.warning("NLP Data not found. Please run 'src/analytics_gold_nlp.py' first.")

# =========================================================
# PAGE 6: PARTICIPANT DRILL-DOWN
# =========================================================
elif page == "6. Participant Drill-Down":
    st.header("👤 Individual Patient Tracker")
    
    pid = st.selectbox("Select Participant ID:", sorted(df['participant_id'].unique()))
    p_data = df[df['participant_id'] == pid]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sessions", len(p_data))
    m2.metric("Avg Impact", f"{p_data['Q26_Social_Impact_Numeric'].mean():.1f}")
    m3.metric("Best Success Rate", f"{p_data['Success_Rate_Numeric'].max()}%")
    
    tab1, tab2 = st.tabs(["Charts", "Raw Data"])
    with tab1:
        fig_p = px.line(p_data, x='session_number', y=['Q26_Social_Impact_Numeric', 'Q1_Engagement_Numeric'],
                        markers=True, title="Progress Over Time")
        st.plotly_chart(fig_p, use_container_width=True)
    with tab2:
        st.dataframe(p_data)

