import streamlit as st
import plotly.express as px
import pandas as pd

def show(nlp_df, kw_df):
    st.title("🧠 Natural Language Processing Insights")
    st.markdown("### Analyzing the 'Why' behind the numbers using RoBERTa NLP.")
    
    if nlp_df.empty:
        st.warning("NLP Data not found. Please run 'src/analytics_gold_nlp.py' first.")
        return

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
    pid = st.selectbox("Select Participant for Narrative:", participants)
    filtered_nlp = nlp_df[nlp_df['participant_id'] == pid]
    
    for _, row in filtered_nlp.iterrows():
        with st.expander(f"Session {row['session_number']} | Theme: {row.get('Theme_specific_situation', 'N/A')} | Sentiment: {row['Sentiment_Label']}"):
            st.markdown(f"**Confidence:** `{row['Sentiment_Score']:.2f}`")
            st.write(row['Master_Text'])