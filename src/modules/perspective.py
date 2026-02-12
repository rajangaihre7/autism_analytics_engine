import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def show(df):
    st.title(" Perspective Triangulation")
    st.markdown("### Inter-rater Reliability: Parent (P) vs. Therapist (T) on Social Impact Score")

    # Check for 'submitted_by' column
    if 'submitted_by' not in df.columns:
        st.error("Column 'submitted_by' not found. Cannot perform comparison.")
        return

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
        st.subheader("Trajectory Comparison")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=comparison_df['session_number'], y=comparison_df['Parent_Score'],
                                      mode='lines+markers', name='Parent (P)', line=dict(color='#2ecc71', width=3)))
        fig_dual.add_trace(go.Scatter(x=comparison_df['session_number'], y=comparison_df['Therapist_Score'],
                                      mode='lines+markers', name='Therapist (T)', line=dict(color='#3498db', width=3, dash='dot')))
        
        fig_dual.update_layout(title="Parent vs. Therapist View", xaxis_title="Session", yaxis_title="Score (0-10)")
        st.plotly_chart(fig_dual, use_container_width=True)
        '''
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
        '''