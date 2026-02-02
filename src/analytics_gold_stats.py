import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_DIR, 'data', 'bronze', 'toy_pal_synthetic_data_bronze.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'gold', 'statistical_results')

def run_statistical_engine():
    print("Starting Gold Layer: Statistical Engine (15 Queries)...")
    
    if not os.path.exists(BRONZE_PATH):
        print("Bronze file not found.")
        return
        
    df = pd.read_excel(BRONZE_PATH, engine='openpyxl')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    # ==========================================
    # GROUP 1: MEASURING EFFICIENCY (Does it work?)
    # ==========================================
    print("\n--- Processing Group 1: Efficiency ---")
    
    # Q1: Social Impact Trend (Correlation Session vs Q26) [cite: 2]
    corr, p_val = stats.pearsonr(df['session_number'], df['social_impact_score_q26'])
    results.append({'ID': 'Q1', 'Group': 'Efficiency', 'Query': 'Social Impact Trend', 'Stat': corr, 'P-Value': p_val, 'Result': 'Significant' if p_val < 0.05 else 'Not Significant'})

    # Q2: Response Time Reduction (First 3 vs Last 3) [cite: 3]
    first_3 = df[df['session_number'] <= 3]['response_time_min_q15'].mean()
    last_3 = df[df['session_number'] >= 12]['response_time_min_q15'].mean()
    pct_decrease = ((first_3 - last_3) / first_3) * 100
    results.append({'ID': 'Q2', 'Group': 'Efficiency', 'Query': 'Response Time Reduction %', 'Stat': pct_decrease, 'P-Value': 'N/A', 'Result': f"{pct_decrease:.1f}% Improvement"})

    # Q3: Distress Frequency (First Half vs Second Half) [cite: 4]
    high_distress_early = len(df[(df['session_number'] <= 7) & (df['distress_boredom_frustration_score_q8'] > 3)])
    high_distress_late = len(df[(df['session_number'] > 7) & (df['distress_boredom_frustration_score_q8'] > 3)])
    results.append({'ID': 'Q3', 'Group': 'Efficiency', 'Query': 'High Distress Count (Early vs Late)', 'Stat': high_distress_early - high_distress_late, 'P-Value': 'N/A', 'Result': f"Dropped from {high_distress_early} to {high_distress_late}"})

    # Q4: Verbal Engagement Growth (Slope for low starters) [cite: 5]
    # Find students who started low (Session 1, Q4 < 3)
    low_starters = df[(df['session_number'] == 1) & (df['verbal_partic_score_q4'] < 3)]['participant_id'].unique()
    low_starter_data = df[df['participant_id'].isin(low_starters)]
    if not low_starter_data.empty:
        slope, _, _, _, _ = stats.linregress(low_starter_data['session_number'], low_starter_data['verbal_partic_score_q4'])
        results.append({'ID': 'Q4', 'Group': 'Efficiency', 'Query': 'Verbal Growth Slope (Low Starters)', 'Stat': slope, 'P-Value': 'N/A', 'Result': f"Avg +{slope:.2f} pts/session"})

    # ==========================================
    # GROUP 2: KEY DRIVERS (Why does it work?)
    # ==========================================
    print("--- Processing Group 2: Key Drivers ---")

    # Q5: Home Application Correlation (Q20 vs Q26) > (Q1 vs Q26)? [cite: 7]
    corr_home, _ = stats.pearsonr(df['applied_learning_q20'], df['social_impact_score_q26'])
    corr_clinic, _ = stats.pearsonr(df['engagement_score_q1'], df['social_impact_score_q26'])
    results.append({'ID': 'Q5', 'Group': 'Drivers', 'Query': 'Home App Correlation Strength', 'Stat': corr_home, 'P-Value': 'N/A', 'Result': 'Stronger' if corr_home > corr_clinic else 'Weaker'})

    # Q6: Personalization Effect (High Q2 vs Low Q2 on Enjoyment Q7) [cite: 8]
    high_pers = df[df['personalization_score_q2'] >= 4]['enjoyment_score_q7']
    low_pers = df[df['personalization_score_q2'] < 4]['enjoyment_score_q7']
    t_stat, p_val_t = stats.ttest_ind(high_pers, low_pers)
    results.append({'ID': 'Q6', 'Group': 'Drivers', 'Query': 'Personalization Impact on Enjoyment', 'Stat': t_stat, 'P-Value': p_val_t, 'Result': 'Significant' if p_val_t < 0.05 else 'Not Significant'})

    # Q7: Theme Understanding vs Generalization (Q18 vs Q22) [cite: 9]
    corr_theme, p_val_theme = stats.pearsonr(df['theme_understand_q18'], df['generalization_q22'])
    results.append({'ID': 'Q7', 'Group': 'Drivers', 'Query': 'Understanding-Generalization Link', 'Stat': corr_theme, 'P-Value': p_val_theme, 'Result': 'Strong Correlation' if corr_theme > 0.6 else 'Moderate/Weak'})

    # Q8: Real-Life Linking Impact (Q25 High -> Spike in Success Q27?) [cite: 10]
    high_link_success = df[df['real_life_link_q25'] >= 4]['success_percentage']
    low_link_success = df[df['real_life_link_q25'] < 4]['success_percentage']
    diff = high_link_success.mean() - low_link_success.mean()
    results.append({'ID': 'Q8', 'Group': 'Drivers', 'Query': 'Success Boost from Real Life Link', 'Stat': diff, 'P-Value': 'N/A', 'Result': f"+{diff:.1f}% Success Rate"})

    # ==========================================
    # GROUP 3: MECHANISMS OF CHANGE
    # ==========================================
    print("--- Processing Group 3: Mechanisms ---")

    # Q9: Personalization-Initiation Link (Q2 vs Q9) [cite: 12]
    corr_pi, p_val_pi = stats.pearsonr(df['personalization_score_q2'], df['interaction_init_q9'])
    results.append({'ID': 'Q9', 'Group': 'Mechanisms', 'Query': 'Personalization -> Initiation', 'Stat': corr_pi, 'P-Value': p_val_pi, 'Result': 'Positive Driver' if corr_pi > 0.5 else 'Weak Link'})

    # Q10: Agency and Confidence (Q11 vs Q21) [cite: 13]
    corr_ac, p_val_ac = stats.pearsonr(df['creativity_score_q11'], df['confidence_potential_q21'])
    results.append({'ID': 'Q10', 'Group': 'Mechanisms', 'Query': 'Creativity -> Confidence', 'Stat': corr_ac, 'P-Value': p_val_ac, 'Result': 'Predictive' if p_val_ac < 0.05 else 'Not Predictive'})

    # Q11: Age Factor (Age vs Response Time Improvement Slope) [cite: 14, 15]
    # Calculate slope for each participant
    slopes = []
    ages = []
    for pid in df['participant_id'].unique():
        p_data = df[df['participant_id'] == pid]
        if len(p_data) > 2:
            s, _, _, _, _ = stats.linregress(p_data['session_number'], p_data['response_time_min_q15'])
            slopes.append(s)
            ages.append(p_data['age'].iloc[0])
    corr_age, p_val_age = stats.pearsonr(ages, slopes)
    results.append({'ID': 'Q11', 'Group': 'Mechanisms', 'Query': 'Age vs Improvement Speed', 'Stat': corr_age, 'P-Value': p_val_age, 'Result': 'Older improves faster' if corr_age < 0 else 'Younger improves faster'})

    # Q12: Gender Responsiveness (Q3 Male vs Female)pp
    m_scores = df[df['gender'] == 'Male']['emotional_conn_score_q3']
    f_scores = df[df['gender'] == 'Female']['emotional_conn_score_q3']
    if len(f_scores) > 0 and len(m_scores) > 0:
        t_gen, p_gen = stats.ttest_ind(m_scores, f_scores)
        res_gen = 'Significant Diff' if p_gen < 0.05 else 'No Gender Diff'
    else:
        t_gen, p_gen, res_gen = 0, 1.0, "Insufficient Data"
    results.append({'ID': 'Q12', 'Group': 'Mechanisms', 'Query': 'Gender Difference in Emotion', 'Stat': t_gen, 'P-Value': p_gen, 'Result': res_gen})

    # ==========================================
    # GROUP 4: PREDICTIVE INSIGHTS
    # ==========================================
    print("--- Processing Group 4: Predictions ---")

    # Q13: Early Predictors (Avg Q1 sessions 1-3 vs Final Q26) [cite: 18]
    early_eng = df[df['session_number'] <= 3].groupby('participant_id')['engagement_score_q1'].mean()
    final_impact = df[df['session_number'] == 14].set_index('participant_id')['social_impact_score_q26']
    # Align data
    aligned = pd.concat([early_eng, final_impact], axis=1).dropna()
    if not aligned.empty:
        corr_pred, p_pred = stats.pearsonr(aligned['engagement_score_q1'], aligned['social_impact_score_q26'])
        results.append({'ID': 'Q13', 'Group': 'Predictions', 'Query': 'Early Engagement Predicts Outcome', 'Stat': corr_pred, 'P-Value': p_pred, 'Result': 'Strong Predictor' if corr_pred > 0.7 else 'Weak Predictor'})

    # Q14: The "Novelty" Effect (Theme Change vs Same Theme) [cite: 19]
    # We need to detect theme changes
    df['prev_theme'] = df.groupby('participant_id')['Theme_specific_situation'].shift(1)
    df['theme_changed'] = df['Theme_specific_situation'] != df['prev_theme']
    
    change_q4 = df[(df['theme_changed'] == True) & (df['session_number'] > 1)]['verbal_partic_score_q4']
    same_q4 = df[(df['theme_changed'] == False) & (df['session_number'] > 1)]['verbal_partic_score_q4']
    
    if len(change_q4) > 0 and len(same_q4) > 0:
        t_nov, p_nov = stats.ttest_ind(change_q4, same_q4)
        results.append({'ID': 'Q14', 'Group': 'Predictions', 'Query': 'Novelty Effect (Theme Change)', 'Stat': t_nov, 'P-Value': p_nov, 'Result': 'Spike Observed' if t_nov > 0 and p_nov < 0.05 else 'No Significant Spike'})

    # Q15: Caregiver Relationship (Q9 vs Q13) [cite: 20]
    # Note: Mapping Q12 (File) to Q13 (Dataset Relationship Impact)
    corr_rel, p_rel = stats.pearsonr(df['interaction_init_q9'], df['relationship_impact_q13'])
    results.append({'ID': 'Q15', 'Group': 'Predictions', 'Query': 'Initiation Improves Relationship', 'Stat': corr_rel, 'P-Value': p_rel, 'Result': 'Correlated' if p_rel < 0.05 else 'Unrelated'})

    # --- SAVE RESULTS ---
    results_df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, 'gold_statistical_answers.csv')
    results_df.to_csv(save_path, index=False)
    
    print("\nStatistical Engine Complete!")
    print(results_df[['ID', 'Query', 'Result', 'P-Value']])
    print(f"\n Detailed Report saved to: {save_path}")

if __name__ == "__main__":
    run_statistical_engine()