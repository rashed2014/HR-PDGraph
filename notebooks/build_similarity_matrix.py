import os
import sqlite3
import pandas as pd
import logging
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from tqdm import tqdm

# --------------------------------------------------
# Setup: Logging and device check
# --------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --------------------------------------------------
# Load SentenceTransformer embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# --------------------------------------------------
# Output directory for similarity matrices
# --------------------------------------------------

output_dir = "../data/similarity_outputs"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Function: Compute similarity between resume and O*NET entities
# --------------------------------------------------

def compute_resume_similarity(resume_row, df_onet, model, category_name, threshold=0.65, batch_size=32):
    """
    Computes semantic similarity between noun phrases in a resume and O*NET entities
    for a specific category (e.g., skills, knowledge). Returns filtered results ≥ threshold.
    """
    similarity_results = []

    # Dynamic column names based on category
    entity_col = f"{category_name}_entity"
    job_title_col = "job_title"
    data_value_col = "data_value"
    soc_code_col = "onetsoc_code"

    # Validate required columns
    for col in [entity_col, job_title_col, data_value_col, soc_code_col]:
        if col not in df_onet.columns:
            logging.error(f"Missing column '{col}' in O*NET data for category: {category_name}")
            return similarity_results

    # Extract info from resume row
    resume_id = resume_row["resume_id"]
    resume_text = resume_row["resume_text"]
    job_title = resume_row[job_title_col]

    # Filter O*NET for job title match
    df_matched = df_onet[df_onet[job_title_col].str.contains(job_title, case=False, na=False, regex=True)]
    if df_matched.empty:
        logging.warning(f"No {category_name} entities found for job: {job_title} (Resume ID: {resume_id})")
        return similarity_results

    # Extract noun phrases
    blob = TextBlob(resume_text)
    noun_phrases = list(set(blob.noun_phrases))
    if not noun_phrases:
        logging.warning(f"No noun phrases found in resume ID {resume_id}")
        return similarity_results

    # Embed noun phrases and O*NET entities
    resume_embeddings = model.encode(noun_phrases, convert_to_numpy=True, batch_size=batch_size)
    entity_list = df_matched[entity_col].drop_duplicates().tolist()
    entity_embeddings = model.encode(entity_list, convert_to_numpy=True, batch_size=batch_size)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(resume_embeddings, entity_embeddings)

    # Filter and record matches
    for i, noun_phrase in enumerate(noun_phrases):
        for j, entity in enumerate(entity_list):
            score = similarity_matrix[i, j]
            if score >= threshold:
                matched_row = df_matched[df_matched[entity_col] == entity].iloc[0]
                similarity_results.append({
                    "resume_id": resume_id,
                    job_title_col: job_title,
                    "noun_phrase": noun_phrase,
                    entity_col: entity,
                    "similarity_score": round(score, 4),
                    data_value_col: matched_row[data_value_col],
                    soc_code_col: matched_row[soc_code_col]
                })

    return similarity_results

# --------------------------------------------------
# Main execution: loop over all O*NET categories
# --------------------------------------------------

def main():
    # Load resume data
    db_path = "../data/annotations_scenario_1/annotations_scenario_1.db"
    conn = sqlite3.connect(db_path)
    query = """
    SELECT r.id AS resume_id, r.resume_text, pj.job_title
    FROM resumes r
    JOIN annotations a ON r.id = a.resume_id
    JOIN predicted_jobs pj ON r.id = pj.resume_id
    WHERE a.rating >= 3;
    """
    df_resumes = pd.read_sql_query(query, conn)
    conn.close()

    # Process each O*NET category file
    onet_dir = "../data/o_net_files"
    for filename in os.listdir(onet_dir):
        if filename.endswith(".csv"):
            category_name = os.path.splitext(filename)[0].lower()
            logging.info(f"🔍 Processing category: {category_name}")

            df_onet = pd.read_csv(os.path.join(onet_dir, filename))

            all_similarity_results = []
            for _, row in tqdm(df_resumes.iterrows(), total=len(df_resumes), desc=f"Matching for {category_name}"):
                results = compute_resume_similarity(row, df_onet, model, category_name)
                all_similarity_results.extend(results)

            # Save results to CSV
            df_similarity = pd.DataFrame(all_similarity_results).drop_duplicates()
            output_path = os.path.join(output_dir, f"resume_{category_name}_similarity_matrix.csv")
            df_similarity.to_csv(output_path, index=False)
            logging.info(f"✅ Saved: {output_path}")

    logging.info("🎉 Done computing similarity for all O*NET categories.")

# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()
