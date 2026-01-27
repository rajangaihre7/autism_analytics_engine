import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_students=30, num_sessions=14):
    """
    Generates a synthetic dataset for the ToyPal Autism Intervention.
    Simulates clinical trajectories based on Severity, Age, and Intervention efficacy.
    """
    
    # Set seed for reproducibility (Crucial for academic validity)
    np.random.seed(42)
    random.seed(42)
    
    data = []

    print(f"🚀 Initializing Data Generation for {num_students} students over {num_sessions} sessions...")

    for student_id in range(101, 101 + num_students):
        
        # --- 1. STATIC PROFILE GENERATION ---
        # Logic: Randomly assign demographics
        age = np.random.randint(4, 16) # Typical age range for this intervention
        gender = np.random.choice(['Male', 'Female'], p=[0.7, 0.3]) # Autism is more diagnosed in males
        severity = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]) # 1=Mild, 3=Severe
        
        # Logic: Set Baselines based on Severity
        # Level 3 students start with lower engagement and higher distress
        base_engagement = 4.0 if severity == 1 else (3.0 if severity == 2 else 2.0)
        base_response_time = 1.0 if severity == 1 else (2.0 if severity == 2 else 3.5)
        
        # Logic: Assign a "Learning Rate" (How fast they improve)
        # Younger kids (age < 8) might have higher plasticity (faster learning)
        learning_rate = 0.5 if age < 8 else 0.4
        if severity == 3: learning_rate *= 0.6 # Severe cases improve slower

        for session in range(1, num_sessions + 1):
            # Progress Factor (0.0 to 1.0) - represents time passing
            progress = session / num_sessions
            
            # --- 2. DYNAMIC SESSION GENERATION ---
            
            # Feature: Story Theme (Random rotation)
            theme = np.random.choice(['Cooking', 'Space', 'Animals', 'School', 'Travel'])
            
            # Q2: Personalization Score (Random variation per session)
            # Some sessions are better personalized than others
            q2_personalization = int(np.clip(np.random.normal(3.8, 1.0), 1, 5))
            
            # --- EMBEDDING CLINICAL CORRELATIONS (The "Why") ---
            
            
            # Correlation 1: Personalization -> Initiation (Q2 -> Q9)
            # Logic: If Personalization is High, Initiation increases
            q9_initiation = int(np.clip(q2_personalization * 0.8 + (progress * 1.5) + np.random.normal(0, 0.5), 1, 5))

            # Correlation 2: Creativity -> Confidence (Q11 -> Q21)
            q11_creativity = int(np.clip(np.random.normal(3, 1) + (progress * 1.5), 1, 5))
            q21_confidence = int(np.clip(q11_creativity * 0.9 + np.random.normal(0, 0.4), 1, 5))

            # Trend 1: Response Time Reduction (Q15)
            # Logic: Decreases over time. 
            current_rt = base_response_time - (session * 0.1 * learning_rate) + np.random.normal(0, 0.2)
            q15_response_time = round(max(0.1, current_rt), 2)

            # Trend 2: Social Impact Growth (Q26)
            # Logic: Increases with Home Application (Q20)
            q20_home_app = int(np.clip(np.random.normal(3, 1) + (progress * 2), 1, 5))
            q26_social = int(np.clip(base_engagement + (q20_home_app * 0.8) + (session * 0.2), 1, 10))

            # Other Metrics
            q1_engagement = int(np.clip(base_engagement + (progress * 1.5) + np.random.normal(0, 0.5), 1, 5))
            q4_verbal = int(np.clip(np.random.normal(3, 1) + (session * 0.3), 1, 10))
            q8_distress = int(np.clip(4.5 - (session * 0.3 * learning_rate), 1, 5)) # Distress goes down
            q18_theme_und = int(np.clip(np.random.normal(3.5, 1) + (progress * 1), 1, 5))
            q22_generalization = int(np.clip(q18_theme_und * 0.7 + (q20_home_app * 0.3), 1, 5))
            
            # Linking to Real Life (Q25) & Success Rate (Q27)
            q25_reallife = int(np.clip(np.random.normal(3, 1), 1, 5))
            q27_success = int(np.clip(q25_reallife * 15 + 20 + (session * 2), 0, 100))

            # Append Row
            data.append({
                'Participant_ID': student_id,
                'Session_Number': session,
                'Date': f"2024-{session:02d}-01", # Dummy Date
                'Age': age,
                'Gender': gender,
                'Severity_Level': severity,
                'Story_Theme': theme,
                'Q1_Engagement': q1_engagement,
                'Q2_Personalization': q2_personalization,
                'Q4_Verbal_Participation': q4_verbal,
                'Q8_Distress': q8_distress,
                'Q9_Initiated_Interactions': q9_initiation,
                'Q11_Creativity': q11_creativity,
                'Q15_Response_Time_Min': q15_response_time,
                'Q18_Theme_Understanding': q18_theme_und,
                'Q20_Home_Application': q20_home_app,
                'Q21_Potential_Confidence': q21_confidence,
                'Q22_Generalization': q22_generalization,
                'Q25_Real_Life_Link': q25_reallife,
                'Q26_Social_Impact_Score': q26_social,
                'Q27_Success_Percentage': q27_success,
                'Notes_Observations': f"Generated note for session {session}" # Placeholder for NLP later
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to "Bronze Layer" (Raw Data)
    output_filename = 'toy_pal_synthetic_data_bronze.csv'
    df.to_csv(output_filename, index=False)
    print(f"✅ Success! Generated {len(df)} rows.")
    print(f"📂 Data saved to: {output_filename}")
    
    return df

# Run the Generator
if __name__ == "__main__":
    df = generate_synthetic_data()
    print("\n--- Data Preview (First 5 Rows) ---")
    print(df.head())