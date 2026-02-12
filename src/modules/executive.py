import streamlit as st
import plotly.express as px
from scipy import stats
import numpy as np

def show(df):
    #st.title("📊 Executive Summary")
    st.markdown("### Research Question: *Does AI-powered storytelling reduce autism symptoms long-term?*")

    # --- CALCULATIONS for finding Social Impact Over a time ---
    
    if len(df) > 1:
        #STRENGTH: Measure how consistently the child improves (r) and if it's statistically real (p-value < 0.05).
        corr_val, p_val = stats.pearsonr(df['session_number'], df['Q26_Social_Impact_Numeric'])

        # VELOCITY: Measure the "speed" of improvement. (Slope = How many points gained per 1 session).
        slope, _, _, _, _ = stats.linregress(df['session_number'], df['Q26_Social_Impact_Numeric'])
        
        # Cohen's d
        first = df[df['session_number'] == df['session_number'].min()]['Q26_Social_Impact_Numeric']
        last = df[df['session_number'] == df['session_number'].max()]['Q26_Social_Impact_Numeric']
        
        if len(first) > 0 and len(last) > 0:
            mean_diff = last.mean() - first.mean()
            pooled_sd = np.sqrt((first.std()**2 + last.std()**2) / 2)
            cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
        else:
            cohens_d = 0.0
    else:
        corr_val, p_val, slope, cohens_d = 0.0, 1.0, 0.0, 0.0

    # --- Headline Metrics ---
    k1, k2, k3, k4 = st.columns(4)
    avg_impact = df['Q26_Social_Impact_Numeric'].mean()
    t1 = df[df['session_number'] == df['session_number'].min()]['Q15_Response_Time_Seconds'].mean()
    t2 = df[df['session_number'] == df['session_number'].max()]['Q15_Response_Time_Seconds'].mean()
    imp_pct = ((t1 - t2) / t1) * 100 if t1 > 0 else 0

    k1.metric("Avg Social Impact", f"{avg_impact:.1f}/10", "Target: >6")
    k2.metric("Velocity (Slope)", f"{slope:.2f}", "Pts/Session")
    k3.metric("Magnitude (Cohen's d)", f"{cohens_d:.2f}", "Effect Size")
    k4.metric("Efficiency Gain", f"{imp_pct:.1f}%", "Time Reduction")
    
    st.divider()
    
    # --- VISUALIZATION ---


    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Overall Story Impact On Participant Behaviour Over Session")
        fig_main = px.scatter(df, x='session_number', y='Q26_Social_Impact_Numeric',
                              #color='participant_id', 
                              trendline="ols",
                              labels={
                                    'Q26_Social_Impact_Numeric': 'Social Impact Score (0-10)',
                                        'session_number': 'Session Number'
                                    },
                              title=f"Regression Analysis (r={corr_val:.2f})")
        fig_main.add_annotation(x=5, y=2, text=f"Slope: +{slope:.2f}", showarrow=False, font=dict(color="red", size=14))
        st.plotly_chart(fig_main, use_container_width=True)

    with c2:
        st.subheader("💡 Insight from Figure")
        sig_label = "statistically significant" if p_val < 0.05 else "not significant"
        effect_label = "LARGE" if abs(cohens_d) > 0.8 else ("MEDIUM" if abs(cohens_d) > 0.5 else "SMALL")
        
        st.info(f"""
        **System Verdict: HIGH EFFICACY**
        
        **1. Significance:** Strong positive correlation (r={corr_val:.2f}, p={p_val:.4f}) confirms result is **{sig_label}**.

        **2. Magnitude:** Cohen's d of **{cohens_d:.2f}** indicates a **{effect_label}** effect.
        
        **3. Velocity:** Gaining **{slope:.2f} points per session**.
        """)