import pandas as pd
from transformers import pipeline
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_DIR, 'data', 'bronze', 'toy_pal_synthetic_data_bronze.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'gold', 'nlp_results')

def run_nlp_engine():
    print("Starting Gold Layer: NLP Engine (RoBERTa Analysis)...")
    
    if not os.path.exists(BRONZE_PATH):
        print(" Bronze file not found.")
        return

    # Load Data
    df = pd.read_excel(BRONZE_PATH, engine='openpyxl')
    
    # 1. SETUP ROBERTA SENTIMENT PIPELINE
    # We use a pre-trained model specifically for emotion/sentiment
    print("Loading RoBERTa model... (Downloads ~500MB first time)")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    # 2. ANALYZE INTERVENTION NOTES
    # We want to see if parent notes are Positive, Neutral, or Negative
    print(" Analyzing Parent Notes...")
    
    # Take a sample to save time (or remove .head(20) for full run)
    sample_notes = df['notes_intervention'].astype(str).tolist()
    
    # Run Inference
    results = sentiment_pipeline(sample_notes)
    
    # Add results back to DataFrame
    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]
    
    # Map RoBERTa labels (LABEL_0=Negative, LABEL_1=Neutral, LABEL_2=Positive)
    label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
    df['sentiment_human'] = df['sentiment_label'].map(label_map)

    # 3. KEYWORD EXTRACTION (Simple Frequency)
    # What techniques are most mentioned in Positive sessions?
    positive_text = " ".join(df[df['sentiment_human'] == 'Positive']['notes_intervention'])
    # Simple word count logic (in real life, we'd use TF-IDF)
    from collections import Counter
    words = [w for w in positive_text.split() if len(w) > 4]
    common_keywords = Counter(words).most_common(10)

    # 4. SAVE RESULTS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    save_path = os.path.join(OUTPUT_DIR, 'gold_nlp_sentiment.csv')
    df[['participant_id', 'session_number', 'notes_intervention', 'sentiment_human', 'sentiment_score']].to_csv(save_path, index=False)
    
    print("\n NLP Analysis Complete!")
    print("Top Keywords in Positive Sessions:", common_keywords)
    print(f"Sentiment Data saved to: {save_path}")

if __name__ == "__main__":
    run_nlp_engine()