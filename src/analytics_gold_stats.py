import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_statistical_engine():
    # --- SETUP PATHS INSIDE FUNCTION TO PREVENT ERRORS ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # Target Input Path (Silver Layer)
    file_path = os.path.join(PROJECT_ROOT, 'data', 'silver', 'After_transformation_Data', 'silver_cleaned.csv')
    
    # Output Path (Gold Layer)
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'gold', 'statistical_results')
    os.makedirs(output_dir, exist_ok=True)

    print("📊 Starting Statistical Engine...")

    # --- 1. LOAD DATA ---
    if not os.path.exists(file_path):
        # Fallback Check
        fallback = os.path.join(PROJECT_ROOT, 'data', 'silver', 'silver_cleaned.csv')
        if os.path.exists(fallback):
            print(f"   ⚠️ Primary path not found. Using fallback: {fallback}")
            file_path = fallback
        else:
            print(f"❌ Error: Could not find 'silver_cleaned.csv'.")
            print(f"   Checked: {file_path}")
            return

    print(f"   - Input: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   - Loaded {len(df)} rows.")
    
    results = []

    # ==========================================
    # GROUP 1: EFFICIENCY (Does it work?)
    # ==========================================
    print("   - Processing Group 1 (Efficiency)...")

    # Q1: Social Impact Trend (Scale 0-10)
    if len(df) > 1:
        corr, p_val = stats.pearsonr(df['session_number'], df['Q26_Social_Impact_Numeric'])
        res_text = 'Significant' if p_val < 0.05 else 'Not Significant'
        results.append({'ID': 'Q1', 'Group': 'Efficiency', 'Query': 'Social Impact Trend', 'Stat': round(corr, 3), 'Result': res_text})

    # Q2: Response Time Reduction
    first_sess = df['session_number'].min()
    last_sess = df['session_number'].max()
    t1 = df[df['session_number'] == first_sess]['Q15_Response_Time_Seconds'].mean()
    t2 = df[df['session_number'] == last_sess]['Q15_Response_Time_Seconds'].mean()
    pct_decrease = ((t1 - t2) / t1) * 100 if t1 > 0 else 0
    results.append({'ID': 'Q2', 'Group': 'Efficiency', 'Query': 'Response Time Reduction %', 'Stat': round(pct_decrease, 1), 'Result': f"{pct_decrease:.1f}% Improvement"})

    # Q3: Distress Frequency (Scale 0-4, >3 is Severe)
    severe_distress = len(df[df['distress_boredom_frustration_score_Q8'] > 3])
    results.append({'ID': 'Q3', 'Group': 'Efficiency', 'Query': 'Severe Distress Incidents', 'Stat': severe_distress, 'Result': f"{severe_distress} Incidents"})

    # Q4: Verbal Engagement Growth (Scale 0-10, <6 is Low)
    low_starters = df[(df['session_number'] == first_sess) & (df['verbal_participation_score_Q4'] < 6)]['participant_id'].unique()
    low_data = df[df['participant_id'].isin(low_starters)]
    
    if len(low_data) > 1:
        slope, _, _, _, _ = stats.linregress(low_data['session_number'], low_data['verbal_participation_score_Q4'])
        results.append({'ID': 'Q4', 'Group': 'Efficiency', 'Query': 'Verbal Growth (Low Starters)', 'Stat': round(slope, 2), 'Result': f"+{slope:.2f} pts/session"})
    else:
        results.append({'ID': 'Q4', 'Group': 'Efficiency', 'Query': 'Verbal Growth (Low Starters)', 'Stat': 0, 'Result': "Insufficient Data (None < 6)"})

    # ==========================================
    # GROUP 2: KEY DRIVERS (Why?)
    # ==========================================
    print("   - Processing Group 2 (Drivers)...")

    # Q5: Home vs Clinic
    corr_home, _ = stats.pearsonr(df['applied_learning_Q20'], df['Q26_Social_Impact_Numeric'])
    corr_clinic, _ = stats.pearsonr(df['Q1_Engagement_Numeric'], df['Q26_Social_Impact_Numeric'])
    winner = 'Home' if abs(corr_home) > abs(corr_clinic) else 'Clinic'
    results.append({'ID': 'Q5', 'Group': 'Drivers', 'Query': 'Stronger Driver', 'Stat': round(max(abs(corr_home), abs(corr_clinic)), 2), 'Result': winner})

    # Q6: Personalization Effect
    # Scale 0-4: We treat >=3 as High
    high_pers = df[df['Q2_Personalization_Numeric'] >= 3]['enjoyment_Q7']
    low_pers = df[df['Q2_Personalization_Numeric'] < 3]['enjoyment_Q7']
    
    if len(high_pers) > 0 and len(low_pers) > 0:
        _, p_val_t = stats.ttest_ind(high_pers, low_pers)
        results.append({'ID': 'Q6', 'Group': 'Drivers', 'Query': 'Personalization Impact', 'Stat': 0, 'Result': 'Significant' if p_val_t < 0.05 else 'Not Significant'})
    else:
        results.append({'ID': 'Q6', 'Group': 'Drivers', 'Query': 'Personalization Impact', 'Stat': 0, 'Result': 'Insufficient Data'})

    # Q7: Understanding vs Generalization
    corr_theme, _ = stats.pearsonr(df['theme_understand_Q18'], df['generalisation_Q22'])
    results.append({'ID': 'Q7', 'Group': 'Drivers', 'Query': 'Understanding-Generalization', 'Stat': round(corr_theme, 2), 'Result': 'Strong' if corr_theme > 0.6 else 'Moderate'})

    # Q8: Real Life Link
    high_link = df[df['real_life_link_Q25'] >= 3]['Success_Rate_Numeric']
    low_link = df[df['real_life_link_Q25'] < 3]['Success_Rate_Numeric']
    if len(high_link) > 0 and len(low_link) > 0:
        diff = high_link.mean() - low_link.mean()
        results.append({'ID': 'Q8', 'Group': 'Drivers', 'Query': 'Success Boost (Real Life Link)', 'Stat': round(diff, 1), 'Result': f"+{diff:.1f}% Success"})
    else:
        results.append({'ID': 'Q8', 'Group': 'Drivers', 'Query': 'Success Boost (Real Life Link)', 'Stat': 0, 'Result': "Insufficient Data"})

    # ==========================================
    # GROUP 3: MECHANISMS (How?)
    # ==========================================
    print("   - Processing Group 3 (Mechanisms)...")

    # Q9: Personalization -> Initiation
    corr_pi, _ = stats.pearsonr(df['Q2_Personalization_Numeric'], df['initiation_Q9'])
    results.append({'ID': 'Q9', 'Group': 'Mechanisms', 'Query': 'Personalization -> Initiation', 'Stat': round(corr_pi, 2), 'Result': 'Positive Driver' if corr_pi > 0.5 else 'Weak Link'})

    # Q10: Creativity -> Confidence
    corr_ac, p_val_ac = stats.pearsonr(df['creativity_Q11'], df['confidence_Q21'])
    results.append({'ID': 'Q10', 'Group': 'Mechanisms', 'Query': 'Creativity -> Confidence', 'Stat': round(corr_ac, 2), 'Result': 'Predictive' if p_val_ac < 0.05 else 'Not Predictive'})

    # Q11: Age vs Improvement Slope
    slopes, ages = [], []
    for pid in df['participant_id'].unique():
        p_data = df[df['participant_id'] == pid]
        if len(p_data) > 1 and p_data['session_number'].nunique() > 1:
            s, _, _, _, _ = stats.linregress(p_data['session_number'], p_data['Q15_Response_Time_Seconds'])
            slopes.append(s)
            ages.append(p_data['age'].iloc[0])
    if len(ages) > 1:
        corr_age, _ = stats.pearsonr(ages, slopes)
        results.append({'ID': 'Q11', 'Group': 'Mechanisms', 'Query': 'Age vs Improvement', 'Stat': round(corr_age, 2), 'Result': 'Older improves faster' if corr_age < 0 else 'Younger improves faster'})
    else:
        results.append({'ID': 'Q11', 'Group': 'Mechanisms', 'Query': 'Age vs Improvement', 'Stat': 0, 'Result': 'Insufficient Data'})

    # Q12: Gender Differences
    m_scores = df[df['gender'] == 'Male']['emotional_connection_Q3']
    f_scores = df[df['gender'] == 'Female']['emotional_connection_Q3']
    if len(m_scores) > 0 and len(f_scores) > 0:
        _, p_gen = stats.ttest_ind(m_scores, f_scores)
        results.append({'ID': 'Q12', 'Group': 'Mechanisms', 'Query': 'Gender Difference', 'Stat': 0, 'Result': 'Significant' if p_gen < 0.05 else 'No Diff'})
    else:
        results.append({'ID': 'Q12', 'Group': 'Mechanisms', 'Query': 'Gender Difference', 'Stat': 0, 'Result': 'One Gender Dominant'})

    # ==========================================
    # GROUP 4: PREDICTIVE INSIGHTS
    # ==========================================
    print("   - Processing Group 4 (Predictions)...")

    # Q13: Early Predictors
    # Grouping by ID to handle potential duplicates (Therapist/Parent) in sessions
    early_eng = df[df['session_number'] <= 3].groupby('participant_id')['Q1_Engagement_Numeric'].mean()
    final_impact = df[df['session_number'] == last_sess].groupby('participant_id')['Q26_Social_Impact_Numeric'].mean()
    
    aligned = pd.concat([early_eng, final_impact], axis=1).dropna()
    
    if len(aligned) > 2:
        corr_pred, p_pred = stats.pearsonr(aligned['Q1_Engagement_Numeric'], aligned['Q26_Social_Impact_Numeric'])
        results.append({'ID': 'Q13', 'Group': 'Predictions', 'Query': 'Early Engagement Predicts Outcome', 'Stat': round(corr_pred, 2), 'Result': 'Strong' if corr_pred > 0.7 else 'Weak'})
    else:
        results.append({'ID': 'Q13', 'Group': 'Predictions', 'Query': 'Early Engagement Predicts Outcome', 'Stat': 0, 'Result': 'Insufficient Data'})

    # Q15: Relationship Impact
    corr_rel, p_rel = stats.pearsonr(df['initiation_Q9'], df['relationship_impact_Q13'])
    results.append({'ID': 'Q15', 'Group': 'Predictions', 'Query': 'Initiation -> Relationship', 'Stat': round(corr_rel, 2), 'Result': 'Correlated' if p_rel < 0.05 else 'Unrelated'})

    # --- SAVE ---
    save_path = os.path.join(output_dir, 'gold_statistical_answers.csv')
    try:
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"✅ DONE! Results saved to: {save_path}")
    except PermissionError:
        print(f"❌ ERROR: Permission Denied. Please close 'gold_statistical_answers.csv' in Excel and try again.")

if __name__ == "__main__":
    run_statistical_engine()