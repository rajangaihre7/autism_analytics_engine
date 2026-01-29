import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# We use relative paths so it works on any computer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_DIR, 'data', 'bronze', 'toy_pal_synthetic_data_bronze.csv')
SILVER_PATH = os.path.join(BASE_DIR, 'data', 'silver', 'toy_pal_analytics_silver.csv')

def process_silver_layer():
    print("Starting Silver Layer Processing")

    # 1. LOAD: Read the raw Bronze data
    if not os.path.exists(BRONZE_PATH):
        print(f"Error: Bronze file not found at {BRONZE_PATH}")
        return

    df = pd.read_csv(BRONZE_PATH)
    print(f"Loaded {len(df)} rows from Bronze.")

    # 2. CLEAN: Deduplication 
   
    initial_count = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate rows.")

    # 3. TRANSFORM: Feature Engineering
    
    # Calculate exact improvement
    df['improvement_score'] = df['post_intervention_score'] - df['pre_intervention_score']
    
    # Calculate percentage improvement (handling division by zero safety)
    df['pct_improvement'] = (df['improvement_score'] / df['pre_intervention_score']) * 100
    df['pct_improvement'] = df['pct_improvement'].round(2)

    # 4. ENRICH: Categorize the results (Business Logic)
    # "High Responder" = Improvement > 15 points
    # "Moderate Responder" = Improvement between 5 and 15
    # "Low/No Responder" = Improvement < 5
    
    conditions = [
        (df['improvement_score'] > 15),
        (df['improvement_score'] >= 5) & (df['improvement_score'] <= 15),
        (df['improvement_score'] < 5)
    ]
    labels = ['High Responder', 'Moderate Responder', 'Low/No Responder']
    
    df['response_category'] = np.select(conditions, labels, default='Unknown')

    # 5. SAVE: Write to Silver folder
    df.to_csv(SILVER_PATH, index=False)
    print(f"Saved Silver Data to: {SILVER_PATH}")
    
    # Preview the first few rows
    print("\n--- Silver Data Preview ---")
    print(df[['patient_id', 'pre_intervention_score', 'post_intervention_score', 'improvement_score', 'response_category']].head())

if __name__ == "__main__":
    process_silver_layer()