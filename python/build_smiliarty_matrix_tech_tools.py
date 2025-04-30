import os
import pandas as pd
import numpy as np
import torch
from textblob import TextBlob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import logging

# -------------------------
# Config
# -------------------------
DATA_DIR = "../data/o_net_tools_tech/"
TOOLS_FILE = "Tools Used.xlsx"
TECH_FILE = "Technology Skills.xlsx"
RESUME_FILE = "../data/annotations_scenario_1/cleaned_resumes.csv"

OUTPUT_DIR = "../data/similarity_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

job_title_json_path = "../data/job_titles/jobs_titles.json"
with open(job_title_json_path, "r") as f:
    job_title_dict = json.load(f)

# Extract the first element (actual job title) from each value
ALLOWED_JOB_TITLES = set(v[0] for v in job_title_dict.values())
logging.info(f"Loaded {len(ALLOWED_JOB_TITLES)} allowed job titles.")

TOOLS_OUTPUT = os.path.join(OUTPUT_DIR, "similarity_matrix_tools.csv")
TECH_OUTPUT = os.path.join(OUTPUT_DIR, "similarity_matrix_tech.csv")
SIMILARITY_THRESHOLD = 0.65

# -------------------------
# Load Data
# -------------------------
print("📥 Loading input files...")
tools_df = pd.read_excel(os.path.join(DATA_DIR, TOOLS_FILE))
tech_df = pd.read_excel(os.path.join(DATA_DIR, TECH_FILE))
resume_df = pd.read_csv(RESUME_FILE)

# Clean and rename relevant columns
tools_df = tools_df.rename(columns={'O*NET-SOC Code': 'onetsoc_code', 'Example': 'tool_example'})
tech_df = tech_df.rename(columns={'O*NET-SOC Code': 'onetsoc_code', 'Example': 'tech_example'})
tools_df = tools_df.dropna(subset=['tool_example'])
tech_df = tech_df.dropna(subset=['tech_example'])





# ✅ Filter to only the relevant job titles
tools_df = tools_df[tools_df['Title'].isin(ALLOWED_JOB_TITLES)]
tech_df = tech_df[tech_df['Title'].isin(ALLOWED_JOB_TITLES)]



# -------------------------
# Load embedding model
# -------------------------
print("🧠 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# -------------------------
# Embed Tools and Tech Examples
# -------------------------
print("⚙️ Encoding Tools and Tech examples...")

tools_df['embedding'] = model.encode(
    tools_df['tool_example'].astype(str).tolist(),
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True
).tolist()

tech_df['embedding'] = model.encode(
    tech_df['tech_example'].astype(str).tolist(),
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True
).tolist()

tools_tensor = torch.nn.functional.normalize(torch.tensor(np.vstack(tools_df['embedding'].values), dtype=torch.float32).to("cuda"), dim=1)
tech_tensor = torch.nn.functional.normalize(torch.tensor(np.vstack(tech_df['embedding'].values), dtype=torch.float32).to("cuda"), dim=1)

# -------------------------
# Matching function
# -------------------------
def match_entities(resume_id, resume_text, noun_phrases, noun_embeddings, entity_df, entity_tensor, entity_type):
    matches = []
    noun_tensor = torch.tensor(np.vstack(noun_embeddings), dtype=torch.float32).to("cuda")
    noun_tensor = torch.nn.functional.normalize(noun_tensor, dim=1)

    sim_matrix = torch.matmul(noun_tensor, entity_tensor.T)
    max_sim, max_idx = sim_matrix.max(dim=1)
    mask = max_sim >= SIMILARITY_THRESHOLD

    for i in torch.where(mask)[0].tolist():
        entity_row = entity_df.iloc[max_idx[i].item()]
        similarity = max_sim[i].item()
        noun_phrase = noun_phrases[i]

        if entity_type == "Tool":
            score = 1.0
            entity_col = "tools_entity"
            example_text = entity_row['tool_example']
        else:
            score = 1.0 if entity_row.get("Hot Technology", "N") == "Y" else 0.75
            entity_col = "tech_entity"
            example_text = entity_row['tech_example']

        match = {
            'resume_id': resume_id,
            'resume_text': resume_text,
            'noun_phrase': noun_phrase,
            entity_col: entity_row['Commodity Title'],
            'entity_text': example_text,
            'similarity_score': similarity,
            'entity_job_title': entity_row['Title'],
            'onetsoc_code': entity_row['onetsoc_code'],
            'scale_id': 'N/A',
            'data_value': score
        }

        matches.append(match)

    return matches

# -------------------------
# Main Matching Loop
# -------------------------
print("🔁 Matching noun phrases for each resume...")
tool_matches = []
tech_matches = []

for _, row in tqdm(resume_df.iterrows(), total=len(resume_df)):
    resume_id = row['resume_id']
    resume_text = row['resume_text']
    noun_phrases = list(set(TextBlob(resume_text).noun_phrases))  # deduplicate

    if not noun_phrases:
        continue

    noun_embeddings = model.encode(noun_phrases, batch_size=32, convert_to_numpy=True)

    # Match to tools
    tool_matches.extend(
        match_entities(resume_id, resume_text, noun_phrases, noun_embeddings, tools_df, tools_tensor, "Tool")
    )

    # Match to tech
    tech_matches.extend(
        match_entities(resume_id, resume_text, noun_phrases, noun_embeddings, tech_df, tech_tensor, "Tech")
    )

# -------------------------
# Save Results
# -------------------------
tool_columns = [
    'resume_id', 'resume_text', 'noun_phrase', 'tools_entity', 'entity_text',
    'similarity_score', 'entity_job_title', 'onetsoc_code', 'scale_id', 'data_value'
]

tech_columns = [
    'resume_id', 'resume_text', 'noun_phrase', 'tech_entity', 'entity_text',
    'similarity_score', 'entity_job_title', 'onetsoc_code', 'scale_id', 'data_value'
]

print(f"💾 Saving outputs to:\n - {TOOLS_OUTPUT}\n - {TECH_OUTPUT}")
pd.DataFrame(tool_matches)[tool_columns].to_csv(TOOLS_OUTPUT, index=False)
pd.DataFrame(tech_matches)[tech_columns].to_csv(TECH_OUTPUT, index=False)
print("✅ Done.")
