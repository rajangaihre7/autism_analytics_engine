import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# --- CONFIGURATION ---
NUM_PARTICIPANTS = 30
MAX_SESSIONS = 14
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'bronze', 'toy_pal_synthetic_data_bronze.xlsx')

# --- CLINICAL VOCABULARY BANKS ---
DIAGNOSES = [
    "Classic Autism (Level 2) + Emotion dysregulation",
    "ASD Level 1 (High Functioning) + Anxiety",
    "ASD Level 2 + Speech Delay",
    "ASD Level 3 + High Support Needs",
    "Pervasive Developmental Disorder (PDD-NOS)"
]

INTERESTS = [
    "Trains and mechanical systems",
    "Dinosaurs and fossils",
    "Solar system and space",
    "Minecraft and building blocks",
    "Disney princesses (Frozen)",
    "Elevators and numbers",
    "Marine biology (Sharks)"
]

THEMES = [
    "Anger Management", "Taking Turns", "Handling Change", "Making Friends",
    "Sensory Overload", "Asking for Help", "Understanding Sarcasm", "Personal Space"
]

THERAPISTS = ["A.Sharifi", "Dr. Emily Chen", "B. Johnson", "S. Patel"]

# --- ADVANCED TEXT GENERATION LOGIC ---

def get_intervention_note(theme, interest, engagement, distress):
    """
    Generates realistic clinical intervention notes.
    Structure: [Action Taken] + [Context/Tool Used] + [Outcome/Reasoning]
    """
    verbs = ["utilized", "employed", "integrated", "scaffolded", "introduced", "modeled", "applied"]
    
    strategies = {
        "Anger Management": ["deep breathing protocols", "a '5-count' grounding technique", "visual 'calm down' maps", "tactile stress-relief objects"],
        "Taking Turns": ["a digital visual timer", "a physical 'turn-taking' baton", "musical cues for transition", "structured role-play scenarios"],
        "Handling Change": ["social stories regarding transitions", "a 'First-Then' visual board", "predictive scheduling", "front-loading of upcoming changes"],
        "Making Friends": ["scripted social greetings", "joint attention activities", "facial emotion recognition cards", "reciprocal play modeling"],
        "Sensory Overload": ["auditory dampening (headphones)", "proprioceptive heavy work", "dimmed lighting environments", "a designated quiet zone"],
        "Asking for Help": ["communication exchange cards (PECS)", "verbal sentence starters", "gestural prompting", "a 'help' button request"]
    }
    
    # Select specific strategy for the theme
    strategy = random.choice(strategies.get(theme, ["standard behavioral prompting"]))
    verb = random.choice(verbs)

    if engagement >= 4:
        templates = [
            f"Therapist {verb} {strategy} by incorporating {interest} characters as peer models.",
            f"Parent successfully {verb} {strategy}; child was highly motivated by the {interest} narrative.",
            f"Session {verb} {strategy} within a customized {interest} storyline, capturing sustained attention.",
            f"The AI {verb} adaptive storytelling to model {strategy} using {interest} analogies.",
            f"Child practiced {strategy} after the parent {verb} {interest}-themed positive reinforcement."
        ]
    elif distress >= 4:
        templates = [
            f"Attempted to {verb} {strategy}, but session was paused due to sensory escalation.",
            f"Parent {verb} {strategy} as a de-escalation measure when the child became overwhelmed.",
            f"Standard instruction failed; therapist {verb} {strategy} using {interest} props to regain composure.",
            f"Child refused the task; {verb} a simplified version of {strategy} to reduce anxiety.",
            f"Due to high distress, the session focused solely on {verb} {strategy} for emotional regulation."
        ]
    else: # Moderate/Low Engagement
        templates = [
            f"Moderately {verb} {strategy} with frequent verbal reminders required.",
            f"Therapist {verb} {strategy} but required hand-over-hand prompting for compliance.",
            f"Parent {verb} visual cues to support the child's understanding of {strategy}.",
            f"Slowly {verb} {strategy}, though the child's attention wavered throughout.",
            f"{verb.capitalize()} {strategy} alongside {interest} visuals to re-engage the child."
        ]
    
    return random.choice(templates)

def get_observation_note(theme, interest, engagement, success_pct):
    """
    Generates realistic behavioral observations.
    Structure: [Behavior Observed] + [Link to Interest/Theme] + [Quantitative Indicator]
    """
    behavior_verbs = ["demonstrated", "exhibited", "displayed", "manifested", "conveyed", "showed"]
    adverbs = ["enthusiastically", "hesitantly", "reluctantly", "consistently", "intermittently"]
    
    verb = random.choice(behavior_verbs)
    adverb = random.choice(adverbs)

    if engagement >= 5:
        templates = [
            f"Child {verb} exceptional mastery of {theme}, explicitly driven by the {interest} content.",
            f"Participant {adverb} engaged with the storyline and {verb} clear retention of the {theme} lesson.",
            f"Zero stimming behaviors observed; child was hyper-focused on the {interest} visual elements.",
            f"Child {verb} the ability to generalize {theme} by spontaneously referencing {interest} characters.",
            f"Spontaneously {verb} the {theme} skill during the post-session interview without prompting."
        ]
    elif engagement <= 2:
        templates = [
            f"Child {verb} signs of fatigue and {adverb} refused to participate in {theme} activities.",
            f"Participant struggled to focus; {verb} minimal interest in the {interest} narrative.",
            f"Frequent wandering observed; child {verb} no retention of the {theme} concepts.",
            f"Session was fragmented; child {verb} high distractibility despite multiple redirection attempts.",
            f"Child {adverb} pushed away the device, indicating strong disinterest in {theme}."
        ]
    else:
        templates = [
            f"Child {verb} inconsistent attention, fluctuating between the {interest} visuals and the task.",
            f"Participant {verb} partial understanding of {theme} but needed significant scaffolding.",
            f"Child {adverb} followed instructions but {verb} a flat affect/lack of enthusiasm.",
            f"Response latency was high; child eventually {verb} the correct behavior after a delay.",
            f"Child {verb} the skill only when directly prompted with {interest} rewards."
        ]

    return random.choice(templates)

# --- CORE GENERATOR LOGIC ---

def generate_participant_profile(p_id):
    return {
        "participant_id": p_id,
        "age": random.randint(5, 12),
        "gender": random.choice(["Male", "Female"]),
        "diagnosis": random.choice(DIAGNOSES),
        "baseline_severity": random.randint(3, 9),
        "special_interest": random.choice(INTERESTS),
        "submitted_type": random.choice(["P", "P", "T"]), # Weighted to Parents
        "therapist_name": random.choice(THERAPISTS)
    }

def simulate_session(profile, session_num, start_date):
    # Dynamic variables
    theme = THEMES[(session_num - 1) % len(THEMES)]
    
    # Engagement Logic (Improvement over time)
    base_eng = 3.0 + (session_num * 0.12) - (profile['baseline_severity'] * 0.1)
    q1_engagement = int(np.clip(base_eng + np.random.normal(0, 0.7), 1, 5))
    
    # Distress Logic (Inverse to engagement)
    q8_distress = int(np.clip(6 - q1_engagement + np.random.normal(0, 0.8), 1, 5))
    
    # Personalization Logic
    q2_personalization = int(np.clip(3.5 + (session_num * 0.1), 1, 5))
    
    # Success Calculation
    opps = random.randint(4, 10)
    success_rate = (q1_engagement / 5.5) + random.uniform(-0.1, 0.1)
    success_count = int(opps * max(0, min(1.0, success_rate)))
    success_pct = round((success_count/opps)*100, 1) if opps > 0 else 0
    
    # Real Life Link (Lagging metric)
    q25_real_life = int(np.clip((q1_engagement * 0.6) + (session_num * 0.15), 1, 5))

    # Generate NLP Notes
    note_int = get_intervention_note(theme, profile['special_interest'], q1_engagement, q8_distress)
    note_obs = get_observation_note(theme, profile['special_interest'], q1_engagement, success_pct)

    return {
        # ID & Context
        "participant_id": profile["participant_id"],
        "session_number": session_num,
        "session_date": (start_date + timedelta(days=session_num*7)).strftime("%Y-%m-%d"),
        "submitted_type": profile["submitted_type"],
        "therapist_parent_name": profile["therapist_name"] if profile["submitted_type"] == "T" else "Parent",
        "age": profile["age"],
        "gender": profile["gender"],
        "diagnosis": profile["diagnosis"],
        "baseline_severity": profile["baseline_severity"],
        "other_information": theme,
        "special_interest": profile["special_interest"],
        "observed_stimming": random.choice(["Hand flapping", "Rocking", "Echolalia", "None"]),
        
        # Metrics (1-5 Scale)
        "engagement_score_q1": q1_engagement,
        "personalization_score_q2": q2_personalization,
        "emotional_conn_score_q3": int(np.clip(q1_engagement + np.random.normal(0, 0.5), 1, 5)),
        "verbal_partic_score_q4": int(np.clip(q1_engagement * 2, 1, 10)),
        "attention_maint_q5": int(np.clip(q1_engagement, 1, 5)),
        "retell_likelihood_q6": int(np.clip(q2_personalization - 1, 1, 5)),
        "enjoyment_score_q7": int(np.clip(q1_engagement + 0.5, 1, 5)),
        "distress_boredom_frustration_score_q8": q8_distress,
        "interaction_init_q9": int(np.clip(q1_engagement - 1, 1, 5)),
        "repetition_score_q10": random.randint(1, 5),
        "creativity_score_q11": int(np.clip(q2_personalization, 1, 5)),
        "relationship_impact_q13": int(np.clip(2 + (session_num * 0.2), 1, 5)),
        "feelings_express_q14": int(np.clip(q1_engagement, 1, 5)),
        "response_time_min_q15": round(max(0.5, 6.0 - (session_num * 0.2)), 2),
        "theme_understand_q18": int(np.clip(q25_real_life + 0.5, 1, 5)),
        "applied_learning_q20": int(np.clip(q25_real_life, 1, 5)),
        "confidence_potential_q21": int(np.clip(q1_engagement, 1, 5)),
        "generalization_q22": int(np.clip(q25_real_life - 0.5, 1, 5)),
        "recall_previous_story_q23": int(np.clip(1 + (session_num * 0.25), 1, 5)),
        "reflect_comment_after_story_ended_q24": random.randint(1, 5),
        "real_life_link_q25": q25_real_life,
        "social_impact_score_q26": int(np.clip(q25_real_life * 2, 1, 10)),
        
        # Qualitative
        "Theme_specific_situation": theme,
        "engagement_opportunities_count": opps,
        "success_count": success_count,
        "success_percentage": success_pct,
        "notes_intervention": note_int,
        "notes_observations": note_obs
    }

def generate_dataset():
    print(f" Generating REALISTIC Clinical Data for NLP (N={NUM_PARTICIPANTS} x {MAX_SESSIONS})...")
    
    all_data = []
    start_date = datetime.now() - timedelta(days=120)

    for i in range(1, NUM_PARTICIPANTS + 1):
        profile = generate_participant_profile(100 + i)
        for s in range(1, MAX_SESSIONS + 1):
            session_data = simulate_session(profile, s, start_date)
            all_data.append(session_data)

    df = pd.DataFrame(all_data)
    
    # Save as EXCEL
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    
    print(f" Success! Generated {len(df)} rows.")
    print(f" Excel Sheet saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dataset()