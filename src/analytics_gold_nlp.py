import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from transformers import pipeline

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'silver', 'After_transformation_Data', 'silver_cleaned.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'gold', 'nlp_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOPWORDS = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'was', 'he', 'his', 'she', 'her', 'it', 'for', 'on', 'with', 'as', 'at', 'this', 'by', 'an', 'nan', 'none', 'story', 'session', 'child'])

def run_nlp_engine():
    print("🧠 Starting Result-Oriented NLP Engine...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # 1. DEFINE CONTEXT & RESULT COLUMNS
    theme_col = 'Theme_specific_situation'
    eng_col = 'Q1_Engagement_Numeric'
    succ_col = 'Success_Rate_Numeric'
    notes_col = 'additional_notes_observations'
    
    # Dynamically find all available comment columns
    comment_cols = sorted([c for c in df.columns if 'comment' in c.lower()])

    print(f"   - Orchestrating narrative: Theme + Engagement + Success + {len(comment_cols)} Comments.")

    # 2. CONTEXT-AWARE NARRATIVE AGGREGATION
    def build_narrative(row):
        # Create the Header: Theme, Engagement, and Success
        header = f"Theme: {row.get(theme_col, 'N/A')}. Result: Engagement Score {row.get(eng_col, 0)}, Success Rate {row.get(succ_col, 0)}%."
        
        # Collect all comments and notes
        body_parts = []
        if pd.notna(row.get(notes_col)):
            body_parts.append(str(row[notes_col]))
        
        for col in comment_cols:
            if pd.notna(row.get(col)) and str(row[col]).strip() != "":
                body_parts.append(f"{col.replace('_', ' ')}: {row[col]}")
        
        # Combine Header and Body
        full_text = f"{header} Details: {' '.join(body_parts)}"
        return full_text

    df['Master_Text'] = df.apply(build_narrative, axis=1)
    
    # Filter for rows with actual content
    df_nlp = df[df['Master_Text'].str.strip() != ""].copy()

    # 3. SENTIMENT ANALYSIS (RoBERTa)
    print("   - Initializing RoBERTa Sentiment Model...")
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    texts = df_nlp['Master_Text'].tolist()
    predictions = sentiment_task(texts, truncation=True, max_length=512)

    # 4. MAPPING
    label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
    df_nlp['Sentiment_Label'] = [label_map[p['label']] for p in predictions]
    df_nlp['Sentiment_Score'] = [p['score'] for p in predictions]

    # 5. KEYWORD TRENDS
    def get_top_keywords(text_series):
        all_words = " ".join(text_series).lower().split()
        clean_words = [re.sub(r'\W+', '', w) for w in all_words if w not in STOPWORDS and len(w) > 3]
        return Counter(clean_words).most_common(20)

    pos_keywords = get_top_keywords(df_nlp[df_nlp['Sentiment_Label'] == 'Positive']['Master_Text'])
    neg_keywords = get_top_keywords(df_nlp[df_nlp['Sentiment_Label'] == 'Negative']['Master_Text'])

    # 6. SAVE OUTPUTS
    sentiment_path = os.path.join(OUTPUT_DIR, 'gold_nlp_full_session_sentiment.csv')
    df_nlp[['participant_id', 'session_number', 'Theme_specific_situation', 'Sentiment_Label', 'Sentiment_Score', 'Master_Text']].to_csv(sentiment_path, index=False)
    
    keywords_df = pd.DataFrame({
        'Positive_Behaviors': [k[0] for k in pos_keywords],
        'Positive_Freq': [k[1] for k in pos_keywords],
        'Negative_Behaviors': [k[0] for k in neg_keywords] if neg_keywords else ["N/A"]*len(pos_keywords),
        'Negative_Freq': [k[1] for k in neg_keywords] if neg_keywords else [0]*len(pos_keywords)
    })
    keywords_df.to_csv(os.path.join(OUTPUT_DIR, 'gold_nlp_keyword_trends.csv'), index=False)

    print(f"✅ NLP Success! Results grouped by Theme and Performance.")

if __name__ == "__main__":
    run_nlp_engine()