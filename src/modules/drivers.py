import streamlit as st
import plotly.express as px
import pandas as pd
from scipy import stats

def show(df, stats_df):
    st.header("🔍 Drivers & Mechanisms")
    
    # --- 1. THE COMPARISON (Home vs. Clinic) ---
    st.subheader("🏆 Critical Analysis: What drives success?")
    st.markdown("Comparing Home Application vs. Story Engagement as predictors of Social Imapact.")

    # A. Calculate Correlations
    # 1. Home Application (Q20)
    if 'applied_learning_Q20' in df.columns:
        r_home, p_home = stats.pearsonr(df['applied_learning_Q20'], df['Q26_Social_Impact_Numeric'])
    else:
        r_home, p_home = 0.0, 1.0

    # 2. Story Engagement (Q1)
    if 'Q1_Engagement_Numeric' in df.columns:
        r_engage, p_engage = stats.pearsonr(df['Q1_Engagement_Numeric'], df['Q26_Social_Impact_Numeric'])
    else:
        r_engage, p_engage = 0.0, 1.0

    # B. Determine the Winner
    winner = "Home Application" if abs(r_home) > abs(r_engage) else "Story Engagement"
    diff = abs(r_home) - abs(r_engage)
    
    # C. Display Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Home Correlation (r)", f"{r_home:.2f}", "Stronger Driver" if winner == "Home Application" else "Weaker")
    c2.metric("Engagement Correlation (r)", f"{r_engage:.2f}", "Stronger Driver" if winner == "Story Engagement" else "Weaker")
    c3.metric("The Winning Driver", f"{winner}", f"By +{abs(diff):.2f} points")

    st.divider()

    # --- 2. VISUALIZATION (Side-by-Side) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Home Practice Effect")
        fig_home = px.scatter(df, x='applied_learning_Q20', y='Q26_Social_Impact_Numeric', 
                              trendline='ols', 
                              title=f"Impact of Home Application (r={r_home:.2f})",
                              labels={'applied_learning_Q20': 'Home Application Score (0-4)', 'Q26_Social_Impact_Numeric': 'Social Impact Score'})
        st.plotly_chart(fig_home, use_container_width=True)

    with col2:
        st.subheader("Story Engagement Effect")
        fig_eng = px.scatter(df, x='Q1_Engagement_Numeric', y='Q26_Social_Impact_Numeric', 
                             trendline='ols', 
                             title=f"Impact of Engagement (r={r_engage:.2f})",
                             labels={'Q1_Engagement_Numeric': 'Engagement Score (0-4)', 'Q26_Social_Impact_Numeric': 'Social Impact Score'})
        st.plotly_chart(fig_eng, use_container_width=True)

    # --- 3. CLINICAL INTERPRETATION ---
    if winner == "Home Application":
        st.success(f"""
        **Clinical Insight:** **Home Application is the primary driver.** The data shows that practicing the lesson at home (r={r_home:.2f}) correlates more strongly with social improvement than simply enjoying the story (r={r_engage:.2f}).
        """)
    else:
        st.info(f"""
        **Clinical Insight:** **Engagement is the primary driver.**
        The data suggests that high engagement (r={r_engage:.2f}) is the most critical factor for success. If the child likes the story, they improve.
        
        """)
'''
#6.	Personalization Effect (Enjoyment): Do sessions with high Personalization Scores
#  (Q2) consistently yield higher Enjoyment Scores (Q7) than sessions with low personalization?

    st.subheader("🎨 The Personalization Premium")
    st.markdown("Does tailoring the story (Q2) actually make it more fun (Q7)?")

    # --- SAFETY CHECK: Ensure columns exist ---
    required_cols = ['Q2_Personalization_Numeric', 'sign_of_enjoyment_Q7']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if not missing_cols:
        # 1. SEGMENT THE DATA (Corrected for 0-4 Scale)
        # High = 3 (Often) or 4 (Very Frequently)
        # Low  = 0, 1, or 2
        df['Personalization_Group'] = df['Q2_Personalization_Numeric'].apply(lambda x: 'High (3-4)' if x >= 3 else 'Low (0-2)')
        
        # 2. CALCULATE STATISTICS (T-Test)
        high_group = df[df['Personalization_Group'] == 'High (3-4)']['sign_of_enjoyment_Q7']
        low_group = df[df['Personalization_Group'] == 'Low (0-2)']['sign_of_enjoyment_Q7']
        
        if len(high_group) > 1 and len(low_group) > 1:
            # Run T-Test
            t_stat, p_val = stats.ttest_ind(high_group, low_group, equal_var=False)
            avg_diff = high_group.mean() - low_group.mean()
            
            # 3. METRICS
            c1, c2, c3 = st.columns(3)
            c1.metric("High Personalization Avg", f"{high_group.mean():.2f}/4")
            c2.metric("Low Personalization Avg", f"{low_group.mean():.2f}/4")
            c3.metric("The 'Premium'", f"+{avg_diff:.2f} pts", "Gain from AI Customization")
            
            # 4. VISUALIZATION (Box Plot with Discrete Axis)
            fig_box = px.box(df, x='Personalization_Group', y='sign_of_enjoyment_Q7',
                             color='Personalization_Group',
                             points="all", # Show the actual dots too!
                             title="Enjoyment Distribution: Tailored vs. Generic",
                             labels={'sign_of_enjoyment_Q7': 'Enjoyment Score (0-4)', 'Personalization_Group': 'Customization Level'},
                             category_orders={'Personalization_Group': ['Low (0-2)', 'High (3-4)']})
            
            # FIX THE Y-AXIS to show 0-4 labels strictly
            fig_box.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=['0: Not at all', '1: Rarely', '2: Occasional', '3: Often', '4: Very Freq'],
                    range=[-0.5, 4.5] # Add padding so dots don't get cut off
                )
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 5. VERDICT
            sig_text = "**statistically significant**" if p_val < 0.05 else "not significant"
            st.info(f"""
            **Research Finding:** Sessions with high personalization (Score 3+) scored **{avg_diff:.2f} points higher** in enjoyment.
            The T-Test (p={p_val:.4f}) confirms this difference is {sig_text}.
            """)
        else:
            st.warning("Not enough data split between High/Low groups to perform statistical comparison.")
            
    else:
        st.error(f"Missing Columns: {missing_cols}")
        '''