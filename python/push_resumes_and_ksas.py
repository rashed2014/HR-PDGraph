import os
import pandas as pd
import logging
from neo4j import GraphDatabase

# ------------------------------------------
# Neo4j connection settings
# ------------------------------------------
NEO4J_URI = "bolt://20.14.162.151:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "recluse2025"

input_dir = "../data/similarity_outputs"
resume_csv_path = "../data/annotations_scenario_1/cleaned_resumes.csv"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------------------
# Step 1: Clear the Neo4j database
# ------------------------------------------

def delete_all_nodes(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def drop_constraints(tx):
    for record in tx.run("SHOW CONSTRAINTS").data():
        tx.run(f"DROP CONSTRAINT {record['name']}")

def drop_indexes(tx):
    for record in tx.run("SHOW INDEXES").data():
        tx.run(f"DROP INDEX {record['name']}")

def clear_database():
    with driver.session() as session:
        logging.info("🧹 Deleting all nodes and relationships...")
        session.execute_write(delete_all_nodes)
    with driver.session() as session:
        logging.info("🧹 Dropping all constraints...")
        session.execute_write(drop_constraints)
    with driver.session() as session:
        logging.info("🧹 Dropping all indexes...")
        session.execute_write(drop_indexes)
    logging.info("✅ Neo4j fully cleared.")

# ------------------------------------------
# Step 2: Create constraints dynamically
# ------------------------------------------

def create_constraints(tx, entity_labels):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resume) REQUIRE r.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:NounPhrase) REQUIRE n.text IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:JobTitle) REQUIRE j.title IS UNIQUE")
    for label in entity_labels:
        tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{label}) REQUIRE e.text IS UNIQUE")
        tx.run(f"CREATE INDEX IF NOT EXISTS FOR (e:{label}) ON (e.onetsoc_code)")

# ------------------------------------------
# Step 3: Load resume CSV and create Resume nodes
# ------------------------------------------

def load_resumes_from_csv():
    df = pd.read_csv(resume_csv_path)
    return df[["resume_id", "resume_text", "original_job"]].drop_duplicates()

def push_resume_node(tx, resume_id, resume_text, original_job):
    tx.run("""
        MERGE (r:Resume {id: $resume_id})
        SET r.resume_text = $resume_text,
            r.original_job = $original_job
    """, resume_id=resume_id, resume_text=resume_text, original_job=original_job)

# ------------------------------------------
# Step 4: Push similarity data and build graph
# ------------------------------------------

def push_to_neo4j(tx, record, entity_label, entity_key):
    query = f"""
        MATCH (r:Resume {{id: $resume_id}})
        MERGE (n:NounPhrase {{text: $noun_phrase}})
        MERGE (e:Entity:{entity_label} {{text: $entity_text}})
        MERGE (j:JobTitle {{title: $job_title}})
        SET j.onetsoc_code = $onetsoc_code
        MERGE (r)-[:CONTAINS]->(n)
        MERGE (n)-[s:SIMILAR_TO]->(e)
        SET s.score = $similarity_score
        {"SET s.example_text = $example_text" if entity_label in ["Tools", "Tech"] else ""}
        MERGE (e)-[rj:REQUIRED_FOR]->(j)
        SET rj.importance = $data_value
    """

    params = {
        "resume_id": record["resume_id"],
        "noun_phrase": record["noun_phrase"],
        "entity_text": record[entity_key],
        "job_title": record["entity_job_title"],
        "similarity_score": round(record["similarity_score"], 4),
        "data_value": record["data_value"],
        "onetsoc_code": record["onetsoc_code"]
    }

    if entity_label in ["Tools", "Tech"]:
        params["example_text"] = record["example_text"]

    tx.run(query, **params)


# ------------------------------------------
# Step 5: Main execution
# ------------------------------------------

def main():
    # Reset the database
    clear_database()

    #resume_abilities_similarity_matrix.csv  resume_skills_similarity_matrix.csv           similarity_matrix_tech.csv
    #resume_knowledge_similarity_matrix.csv  resume_work_activities_similarity_matrix.csv  similarity_matrix_tools.csv

    # Define fixed entity labels and expected files
    entity_files = {
        "Abilities": "resume_abilities_similarity_matrix.csv",
        "Knowledge": "resume_knowledge_similarity_matrix.csv",
        "Skills": "resume_skills_similarity_matrix.csv ",
        "Workactivities": "resume_work_activities_similarity_matrix.csv",
        "Tools": "similarity_matrix_tools.csv",
        "Tech": "similarity_matrix_tech.csv"
    }

    entity_labels = list(entity_files.keys())

    # Create constraints
    with driver.session() as session:
        session.execute_write(create_constraints, entity_labels)
        logging.info(f"✅ Constraints created for: {entity_labels}")

    # Create Resume nodes
    df_resumes = load_resumes_from_csv()
    with driver.session() as session:
        for _, row in df_resumes.iterrows():
            session.execute_write(push_resume_node, row["resume_id"], row["resume_text"], row["original_job"])
    logging.info(f"✅ Created {len(df_resumes)} Resume nodes.")

    # Load similarity CSVs and build graph
    with driver.session() as session:
        for label, file in entity_files.items():
            file_path = os.path.join(input_dir, file)
            if not os.path.exists(file_path):
                logging.warning(f"⚠️ File not found: {file_path}, skipping.")
                continue

            logging.info(f"📂 Loading: {file_path}")
            df = pd.read_csv(file_path)

            entity_key = f"{label.lower()}_entity"
            required_cols = [
                "resume_id", "noun_phrase", entity_key,
                "similarity_score", "data_value", "entity_job_title", "onetsoc_code"
            ]

            if label in ["Tools", "Tech"]:
                required_cols.append("example_text")

            if not all(col in df.columns for col in required_cols):
                logging.warning(f"⚠️ Required columns missing in {file}, skipping.")
                continue

            for _, row in df.iterrows():
                session.execute_write(push_to_neo4j, row, label, entity_key)

            logging.info(f"✅ Finished pushing {label} data.")

    driver.close()
    logging.info("🎉 All graph data pushed successfully to Neo4j.")

# ------------------------------------------
# Run the script
# ------------------------------------------
if __name__ == "__main__":
    main()
