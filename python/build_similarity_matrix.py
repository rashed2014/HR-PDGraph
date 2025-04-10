import os
import pandas as pd
import logging
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
# Compute similarity between resume noun phrases and O*NET entities
# --------------------------------------------------

def compute_resume_similarity(resume_row, df_onet, model, category_name, threshold=0.65, batch_size=32):
    results = []

    entity_col = f"{category_name}_entity"
    required_cols = [entity_col, "onetsoc_code", "job_title", "data_value"]
    for col in required_cols:
        if col not in df_onet.columns:
            logging.error(f"Missing column '{col}' in O*NET data for category '{category_name}'")
            return results
    
     # ❌ Remove rows that are not Importance scale (e.g., Level "LV")
    df_onet = df_onet[df_onet["scale_id"] == "IM"].copy()
    resume_id = resume_row["resume_id"]
    resume_text = resume_row["resume_text"]
    original_job = resume_row["original_job"]

    blob = TextBlob(resume_text)
    noun_phrases = list(set(blob.noun_phrases))
    if not noun_phrases:
        logging.warning(f"No noun phrases found in resume ID {resume_id}")
        return results

    # Unique entity values for embedding
    unique_entities = df_onet[entity_col].dropna().unique().tolist()
    entity_embeddings = model.encode(unique_entities, convert_to_numpy=True, batch_size=batch_size)
    resume_embeddings = model.encode(noun_phrases, convert_to_numpy=True, batch_size=batch_size)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(resume_embeddings, entity_embeddings)

    # Map from entity to all rows with that entity
    grouped_onet = df_onet.groupby(entity_col)

    for i, noun_phrase in enumerate(noun_phrases):
        for j, entity in enumerate(unique_entities):
            score = similarity_matrix[i, j]
            if score >= threshold:
                matched_rows = grouped_onet.get_group(entity)
                for _, matched_row in matched_rows.iterrows():
                    results.append({
                        "resume_id": resume_id,
                        "resume_text": resume_text,
                        "noun_phrase": noun_phrase,
                        f"{category_name}_entity": entity,
                        "similarity_score": round(score, 4),
                        "entity_job_title": matched_row["job_title"],
                        "onetsoc_code": matched_row["onetsoc_code"],
                        "scale_id": matched_row["scale_id"],
                        "data_value": matched_row["data_value"]
                    })

    return results

# --------------------------------------------------
# Main: Process all O*NET category files
# --------------------------------------------------

def main():
    resume_path = "../data/annotations_scenario_1/cleaned_resumes.csv"
    df_resumes = pd.read_csv(resume_path)
    df_resumes = df_resumes[df_resumes["annotation_1"].astype(int) >= 3]

    onet_dir = "../data/o_net_files"
    for filename in os.listdir(onet_dir):
        if filename.endswith(".csv"):
            category = os.path.splitext(filename)[0].lower()
            logging.info(f"🔍 Processing category: {category}")

            df_onet = pd.read_csv(os.path.join(onet_dir, filename))

            all_results = []
            for _, row in tqdm(df_resumes.iterrows(), total=len(df_resumes), desc=f"Matching for {category}"):
                matches = compute_resume_similarity(row, df_onet, model, category)
                all_results.extend(matches)

            df_out = pd.DataFrame(all_results)
            out_path = os.path.join(output_dir, f"resume_{category}_similarity_matrix.csv")
            df_out.to_csv(out_path, index=False)
            logging.info(f"✅ Saved: {out_path}")

    logging.info("🎉 Done computing similarity for all O*NET categories.")

# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
