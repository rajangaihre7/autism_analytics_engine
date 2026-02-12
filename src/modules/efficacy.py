import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# --- HELPER FUNCTION ---
def get_stat_text(stats_df, qid):
    """Helper to fetch stat result safely"""
    if not stats_df.empty and 'ID' in stats_df.columns:
        row = stats_df[stats_df['ID'] == qid]
        if not row.empty:
            return f"**💡 Research Verdict:** {row.iloc[0]['Result']} (Stat: {row.iloc[0]['Stat']})"
    return "💡 Calculation pending..."

# --- MAIN SHOW FUNCTION ---
def show(df, stats_df):
    st.header("🧪 Efficiency & Safety")
    
    # --- 1. METRICS ROW (Learning Speed) ---
    # Calculate Response Time Drop (First 3 vs Last 3)
    all_sessions = sorted(df['session_number'].unique())
    if len(all_sessions) >= 2:
        split_idx = min(3, len(all_sessions) // 2)
        first_block = df[df['session_number'].isin(all_sessions[:split_idx])]['Q15_Response_Time_Seconds']
        last_block = df[df['session_number'].isin(all_sessions[-split_idx:])]['Q15_Response_Time_Seconds']
        
        # Calculate % Drop & Cohen's d
        avg_start = first_block.mean()
        avg_end = last_block.mean()
        pct_drop = ((avg_start - avg_end) / avg_start) * 100 if avg_start > 0 else 0
        
        pooled_sd = np.sqrt((first_block.std()**2 + last_block.std()**2) / 2)
        cohens_d = (avg_start - avg_end) / pooled_sd if pooled_sd > 0 else 0.0
        effect_label = "Large" if abs(cohens_d) > 0.8 else ("Medium" if abs(cohens_d) > 0.5 else "Small")
    else:
        pct_drop, cohens_d, effect_label = 0.0, 0.0, "N/A"

    # Display Metrics
    #st.subheader("⚡ Speed of Adaptation (Learning Curve)")
    #m1, m2, m3 = st.columns(3)
    #m1.metric("Response Time Drop", f"{pct_drop:.1f}%", "First 3 vs. Last 3 Sessions")
    #m2.metric("Effect Size (Cohen's d)", f"{cohens_d:.2f}", f"{effect_label} Improvement")
    #m3.metric("Current Avg Time", f"{avg_end:.1f} sec", "Latest Performance")
    
    #st.divider()

    # --- 2. VISUALIZATION COLUMNS ---
    c1, c2 = st.columns(2)
    
    # COLUMN 1: RESPONSE TIME (Learning Curve)
    with c1:
        st.subheader("Response Time Trajectory")
        time_trend = df.groupby('session_number')['Q15_Response_Time_Seconds'].mean().reset_index()
        
        fig_time = px.line(time_trend, x='session_number', y='Q15_Response_Time_Seconds', markers=True,
                           labels={'Q15_Response_Time_Seconds': 'Avg Response Time (Seconds)', 'session_number': 'Session'},
                           title="Learning Curve: Time to Respond")
        
        # Reverse Y-axis (Lower Time = Better)
        fig_time.update_layout(yaxis_autorange="reversed") 
        st.plotly_chart(fig_time, use_container_width=True)
        
        st.info(f"**Analysis:** The child became **{pct_drop:.1f}% faster on responsing to a emotion or question**") #, showing a **{effect_label}** clinical improvement (d={cohens_d:.2f}) .")
       # st.caption(get_stat_text(stats_df, "Q2"))

   # COLUMN 2: SAFETY & DISTRESS (Adverse Events)
    with c2:
        st.subheader("Showing Distress, Boredom & Frustration")
        
        # --- 1. DEFINE MAPPING ---
        # 0=Not at all, 1=Rarely, 2=Occasionally, 3=Often, 4=Very Frequently
        # Clinical Threshold: We consider "Often" (3) and "Very Frequently" (4) as High Distress.
        
        high_distress_count = len(df[df['distress_boredom_frustration_score_Q8'] >= 3])
        total_sessions = len(df)
        safety_rate = ((total_sessions - high_distress_count) / total_sessions) * 100 if total_sessions > 0 else 0
        
        # --- 2. VISUALIZATION ---
        fig_safe = px.scatter(df, x='session_number', y='distress_boredom_frustration_score_Q8',
                             # color='participant_id',
                              title="Distress Frequency Over Time",
                              labels={'session_number': 'Session'
                                      ,'distress_boredom_frustration_score_Q8': 'distress_boredom_frustration_score'},
                              range_y=[-0.5, 4.5]) # Fix range 0-4
        
        # Custom Y-Axis Labels (The key update!)
        fig_safe.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4],
                ticktext=['Not at all', 'Rarely', 'Occasionally', 'Often', 'Very Freq']
            )
        )
        
        # Add Red Zone for High Distress (Scores 3 & 4)
        fig_safe.add_hrect(y0=2.5, y1=4.5, line_width=0, fillcolor="red", opacity=0.1, 
                           annotation_text="Adverse Zone (Often/Freq)", annotation_position="top left")
        
        st.plotly_chart(fig_safe, use_container_width=True)
        
        # --- 3. CLINICAL VERDICT ---
        if high_distress_count == 0:
            st.success(f" **Protocol Safe:** 100% Safety Rate. No participants reported 'Often' or 'Very Frequent' distress.")
        else:
            st.warning(f" **Attention:** {high_distress_count} sessions reported distress levels of 'Often' (3) or higher. Safety Rate: {safety_rate:.1f}%")
            
       # st.caption(get_stat_text(stats_df, "Q3"))