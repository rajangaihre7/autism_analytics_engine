import pandas as pd
import numpy as np
import os
import re

# --- CONFIGURATION ---
# We use relative paths so it works on any computer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# INPUT: The raw file you just uploaded
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'bronze', 'data_bronze_raw.csv')
# OUTPUT: The clean Master File
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'silver', 'After_transformation_Data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'silver_cleaned.csv')

def clean_response_time(val):
    """ 
    Converts mixed time formats into raw Seconds (Float).
    Examples: '120 seconds' -> 120.0, '2 Minutes' -> 120.0
    """
    if pd.isna(val): return np.nan
    val_str = str(val).lower()
    
    # Extract the numeric part using Regex
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    if not nums: return np.nan
    number = float(nums[0])

    # Convert Logic
    if 'minute' in val_str:
        return number * 60  # Convert minutes to seconds
    # Default assumption is seconds (based on your dataset)
    return number

def clean_percentage(val):
    """ Converts '80%', '60 %' to float 80.0 """
    if pd.isna(val): return np.nan
    val_str = str(val).replace('%', '').strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def run_cleaning_pipeline():
    print(f" Starting Data Cleaning Pipeline...")
    print(f"   - Input: {INPUT_FILE}")

    # 1. LOAD DATA
    if not os.path.exists(INPUT_FILE):
        # Fallback for local testing if folder structure isn't perfect
        if os.path.exists('data_bronze_raw.csv'):
            df = pd.read_csv('data_bronze_raw.csv')
        else:
            print(f" Error: Input file not found!")
            return None
    else:
        df = pd.read_csv(INPUT_FILE)
    
    print(f"   - Loaded {len(df)} rows.")

    # 2. DROP INVALID ROWS
    # Rows without a participant ID are useless
    df = df.dropna(subset=['participant_id'])

    # 3. APPLY CLEANING FUNCTIONS
    # Clean Response Time (The most critical fix)
    # Note: Column name in your update is 'response_time_min_Q15', but data is often in seconds
    if 'response_time_min_Q15' in df.columns:
        df['Q15_Response_Time_Seconds'] = df['response_time_min_Q15'].apply(clean_response_time)
    else:
        print("⚠️ Warning: 'response_time_min_Q15' column missing!")

    # Clean Success Percentage
    if 'success_percentage' in df.columns:
        df['Success_Rate_Numeric'] = df['success_percentage'].apply(clean_percentage)

    # 4. STANDARDIZE COLUMN NAMES
    # This maps your specific CSV headers to the standard names our Analytics Engine expects
    column_mapping = {
        'engagement_score_Q1': 'Q1_Engagement_Numeric',
        'story_personalised_to_participant_Q2': 'Q2_Personalization_Numeric',
        'how_much_different_scenarios_stories_impact_overall_social_behaviour_Q26': 'Q26_Social_Impact_Numeric',
        'distress_boredom_frustration_score_Q8': 'distress_boredom_frustration_score_Q8',
        'verbal_participation_score_Q4': 'verbal_participation_score_Q4',
        'applied_learning_during/immediately after session_both_P &T_Q20': 'applied_learning_Q20',
        'what_extend_participant_understand_theme_Q18': 'theme_understand_Q18',
        'generalise_behaviour_outside_story_Q22': 'generalisation_Q22',
        'Link_story_to_real_life_experiences_Q25': 'real_life_link_Q25',
        'participant_initiate_interaction_Q9': 'initiation_Q9',
        'participant_try_creatively_changes_story_Q11': 'creativity_Q11',
        'participant_feel_confidence&has_potential_appy_story_after_session_Q21': 'confidence_Q21',
        'demonstrate_emotional_connection_Q3': 'emotional_connection_Q3',
        'how_much_relationship_between_particiant& carer/parent Improved_Q13': 'relationship_impact_Q13',
        'sign_of_enjoyment_Q7': 'enjoyment_Q7'
    }

    # Perform the renaming (Create new columns, keep old ones just in case)
    for original, new_name in column_mapping.items():
        if original in df.columns:
            # Force numeric (coerce errors) to handle any stray text
            df[new_name] = pd.to_numeric(df[original], errors='coerce').fillna(0)

    # 5. SAVE GOLD MASTER
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"SUCCESS! Clean Master File saved to:")
    print(f"   {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    run_cleaning_pipeline()