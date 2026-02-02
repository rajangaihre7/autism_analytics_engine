import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ToyPal Research Analytics", layout="wide")

# --- DIAGNOSTIC BLOCK (DELETE THIS LATER) ---
st.title("🛠️ Debugging Mode")
st.write("Current Working Directory:", os.getcwd())
st.write("Files in Current Directory:", os.listdir())

# Check if 'data' folder exists
if os.path.exists('data'):
    st.write("✅ Found 'data' folder.")
    st.write("Contents of 'data':", os.listdir('data'))
    if os.path.exists('data/bronze'):
        st.write("✅ Found 'data/bronze' folder.")
        st.write("Contents:", os.listdir('data/bronze'))
    else:
        st.error("❌ 'data/bronze' NOT found!")
else:
    st.error("❌ 'data' folder NOT found in root directory!")
# -------------------------------------------

# --- REAL APP LOGIC ---
BASE_DIR = os.getcwd() # Simpler path logic for Cloud

@st.cache_data
def load_data():
    # 1. Load Raw Session Data (Bronze)
    # Note the use of simple relative paths
    bronze_path = 'data/bronze/toy_pal_synthetic_data_bronze.xlsx'
    
    if os.path.exists(bronze_path):
        try:
            df = pd.read_excel(bronze_path, engine='openpyxl')
            st.success(f"Successfully loaded Bronze data: {len(df)} rows")
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return None, None, None
    else:
        st.error(f"❌ File not found at: {bronze_path}")
        return None, None, None

    # 2. Load Statistical Answers (Gold)
    stats_path = 'data/gold/statistical_results/gold_statistical_answers.csv'
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
    else:
        stats_df = pd.DataFrame()

    # 3. Load NLP Results (Gold)
    nlp_path = 'data/gold/nlp_results/gold_nlp_sentiment.csv'
    if os.path.exists(nlp_path):
        nlp_df = pd.read_csv(nlp_path)
    else:
        nlp_df = pd.DataFrame()

    return df, stats_df, nlp_df

df, stats_df, nlp_df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Research Modules")
page = st.sidebar.radio("Navigate Findings:", [
    "Overview", 
    "Group 1: Efficiency (Does it work?)", 
    "Group 2: Key Drivers (Why?)", 
    "Group 3: Mechanisms (How?)", 
    "Group 4: Predictive Insights",
    "NLP & Qualitative Analysis"  # <-- NEW PAGE
])

# --- HELPER: Display Stat Result ---
def show_stat_result(qid):
    if not stats_df.empty:
        row = stats_df[stats_df['ID'] == qid]
        if not row.empty:
            res = row.iloc[0]['Result']
            pval = row.iloc[0]['P-Value']
            st.info(f"**Statistical Verdict ({qid}):** {res} (p={pval})")

# --- PAGE: OVERVIEW ---
if page == "Overview":
    st.title("ToyPal Intervention: Dissertation Dashboard")
    st.markdown("### Research Question: Can AI-driven storytelling improve social skills in children with ASD?")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Participants", df['participant_id'].nunique())
    kpi2.metric("Total Sessions", len(df))
    kpi3.metric("Avg Social Impact", round(df['social_impact_score_q26'].mean(), 2))
    if not nlp_df.empty:
        pos_pct = round((len(nlp_df[nlp_df['sentiment_human']=='Positive']) / len(nlp_df)) * 100, 1)
        kpi4.metric("Positive Parent Feedback", f"{pos_pct}%")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Participant Demographics")
        fig_demo = px.pie(df, names='diagnosis', hole=0.4)
        st.plotly_chart(fig_demo, use_container_width=True)
    with c2:
        st.subheader("Sessions per Interest")
        fig_int = px.bar(df['special_interest'].value_counts(), orientation='h')
        st.plotly_chart(fig_int, use_container_width=True)

# --- PAGE: GROUP 1 (EFFICIENCY) ---
elif page == "Group 1: Efficiency (Does it work?)":
    st.header("Group 1: Measuring Efficiency")
    
    st.subheader("Q1: Social Impact Trend")
    trend = df.groupby('session_number')['social_impact_score_q26'].mean().reset_index()
    fig1 = px.line(trend, x='session_number', y='social_impact_score_q26', markers=True, 
                   title="Improvement in Social Impact Scores (1-14)")
    st.plotly_chart(fig1, use_container_width=True)
    show_stat_result("Q1")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Q2: Response Time")
        early = df[df['session_number'] <= 3]['response_time_min_q15'].mean()
        late = df[df['session_number'] >= 12]['response_time_min_q15'].mean()
        fig2 = go.Figure([go.Bar(x=['Initial (Ses 1-3)', 'Final (Ses 12-14)'], y=[early, late], marker_color=['#FFA07A', '#98FB98'])])
        fig2.update_layout(yaxis_title="Minutes", title="Reduction in Response Time")
        st.plotly_chart(fig2, use_container_width=True)
        show_stat_result("Q2")
    
    with col2:
        st.subheader("Q3: Distress Frequency")
        high_distress = df[df['distress_boredom_frustration_score_q8'] > 3].groupby('session_number').size().reset_index(name='count')
        fig3 = px.bar(high_distress, x='session_number', y='count', title="Count of High Distress Incidents per Session")
        st.plotly_chart(fig3, use_container_width=True)
        show_stat_result("Q3")

# --- PAGE: GROUP 2 (DRIVERS) ---
elif page == "Group 2: Key Drivers (Why?)":
    st.header("Group 2: Identifying Key Drivers")

    st.subheader("Q5: Home vs. Clinic Correlation")
    c1, c2 = st.columns(2)
    with c1:
        fig_home = px.scatter(df, x='applied_learning_q20', y='social_impact_score_q26', trendline="ols", title="Home Application (Q20)")
        st.plotly_chart(fig_home, use_container_width=True)
    with c2:
        fig_clinic = px.scatter(df, x='engagement_score_q1', y='social_impact_score_q26', trendline="ols", title="Clinic Engagement (Q1)")
        st.plotly_chart(fig_clinic, use_container_width=True)
    show_stat_result("Q5")

    st.subheader("Q8: The 'Real-Life Link' Boost")
    # Histogram of success rates split by Link score
    fig8 = px.histogram(df, x='success_percentage', color='real_life_link_q25', nbins=20, 
                        title="Success % Distribution by Real-Life Link Score")
    st.plotly_chart(fig8, use_container_width=True)
    show_stat_result("Q8")

# --- PAGE: GROUP 3 (MECHANISMS) ---
elif page == "Group 3: Mechanisms (How?)":
    st.header("Group 3: Mechanisms of Change")

    st.subheader("Q9: Personalization vs. Initiation Heatmap")
    fig9 = px.density_heatmap(df, x='personalization_score_q2', y='interaction_init_q9', 
                              text_auto=True, title="Does Personalization drive Initiation?")
    st.plotly_chart(fig9, use_container_width=True)
    show_stat_result("Q9")

    st.subheader("Q12: Gender Differences in Emotional Connection")
    fig12 = px.box(df, x='gender', y='emotional_conn_score_q3', color='gender', 
                   title="Emotional Connection Scores by Gender")
    st.plotly_chart(fig12, use_container_width=True)
    show_stat_result("Q12")

# --- PAGE: GROUP 4 (PREDICTIVE) ---
elif page == "Group 4: Predictive Insights":
    st.header("Group 4: Predictive Insights")
    
    st.subheader("Q13: Early Predictor Analysis")
    # Prepare data again for plot
    early_eng = df[df['session_number'] <= 3].groupby('participant_id')['engagement_score_q1'].mean()
    final_impact = df[df['session_number'] == 14].set_index('participant_id')['social_impact_score_q26']
    pred_data = pd.concat([early_eng, final_impact], axis=1).dropna()
    pred_data.columns = ['Early_Engagement', 'Final_Impact']
    
    fig13 = px.scatter(pred_data, x='Early_Engagement', y='Final_Impact', trendline='ols',
                       title="Can we predict Final Impact from First 3 Sessions?")
    st.plotly_chart(fig13, use_container_width=True)
    show_stat_result("Q13")

# --- PAGE: NLP ANALYSIS (NEW) ---
elif page == "NLP & Qualitative Analysis":
    st.header("🧠 Natural Language Processing (NLP) Insights")
    
    if nlp_df.empty:
        st.warning("NLP Results not found. Run 'analytics_gold_nlp.py' first.")
    else:
        # 1. Sentiment Overview
        st.subheader("Parent Feedback Sentiment Analysis (RoBERTa Model)")
        sent_counts = nlp_df['sentiment_human'].value_counts()
        fig_nlp1 = px.pie(values=sent_counts.values, names=sent_counts.index, 
                          color=sent_counts.index,
                          color_discrete_map={'Positive':'#2E8B57', 'Neutral':'#F0E68C', 'Negative':'#CD5C5C'},
                          title="Overall Sentiment Distribution of Intervention Notes")
        st.plotly_chart(fig_nlp1, use_container_width=True)

        # 2. Drill Down
        st.subheader("Sentiment by Session Number")
        # Stacked bar chart of sentiment over time
        sent_time = nlp_df.groupby(['session_number', 'sentiment_human']).size().reset_index(name='count')
        fig_nlp2 = px.bar(sent_time, x='session_number', y='count', color='sentiment_human',
                          color_discrete_map={'Positive':'#2E8B57', 'Neutral':'#F0E68C', 'Negative':'#CD5C5C'},
                          title="Evolution of Parent Sentiment Over 14 Sessions")
        st.plotly_chart(fig_nlp2, use_container_width=True)

        # 3. Raw Text Explorer
        st.subheader("Qualitative Note Explorer")
        filter_sent = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Negative", "Neutral"])
        
        if filter_sent != "All":
            filtered_notes = nlp_df[nlp_df['sentiment_human'] == filter_sent]
        else:
            filtered_notes = nlp_df
            
        st.dataframe(filtered_notes[['session_number', 'notes_intervention', 'sentiment_score']].head(10), use_container_width=True)