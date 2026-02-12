import streamlit as st
import plotly.express as px
import pandas as pd

def show(df):
    st.header("👤 Individual Patient Tracker")
    
    # Select Participant
    if 'participant_id' in df.columns:
        participants = sorted(df['participant_id'].unique())
        pid = st.selectbox("Select Participant ID:", participants)
        
        # Filter Data
        p_data = df[df['participant_id'] == pid]
        
        # Mini Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sessions", len(p_data))
        m2.metric("Avg Impact", f"{p_data['Q26_Social_Impact_Numeric'].mean():.1f}")
        m3.metric("Best Success Rate", f"{p_data['Success_Rate_Numeric'].max()}%")
        
        # Tabs for Charts vs Data
        tab1, tab2 = st.tabs(["Charts", "Raw Data"])
        
        with tab1:
            st.subheader("Progress Over Time")
            # Dual Line Chart
            fig_p = px.line(p_data, x='session_number', y=['Q26_Social_Impact_Numeric', 'Q1_Engagement_Numeric'],
                            markers=True, title="Impact vs. Engagement")
            st.plotly_chart(fig_p, use_container_width=True)
            
        with tab2:
            st.dataframe(p_data)
            
    else:
        st.error("⚠️ Dataset missing 'participant_id' column.")