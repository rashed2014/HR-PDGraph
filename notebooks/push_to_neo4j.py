import os
import sqlite3
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
db_path = "../data/annotations_scenario_1/annotations_scenario_1.db"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------------------
# Step 1: Clear the Neo4j database
# ------------------------------------------

def delete_all_nodes(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def drop_constraints(tx):
    constraints = tx.run("SHOW CONSTRAINTS").data()
    for record in constraints:
        tx.run(f"DROP CONSTRAINT {record['name']}")

def drop_indexes(tx):
    indexes = tx.run("SHOW INDEXES").data()
    for record in indexes:
        tx.run(f"DROP INDEX {record['name']}")

def clear_database():
    with driver.session() as session:
        print("🧹 Deleting all nodes and relationships...")
        session.execute_write(delete_all_nodes)

    with driver.session() as session:
        print("🧹 Dropping all constraints...")
        session.execute_write(drop_constraints)

    with driver.session() as session:
        print("🧹 Dropping all indexes...")
        session.execute_write(drop_indexes)

    print("✅ Neo4j fully cleared.")

# ------------------------------------------
# Step 2: Create constraints dynamically
# ------------------------------------------

def create_constraints(tx, entity_labels):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resume) REQUIRE r.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:NounPhrase) REQUIRE n.text IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:JobTitle) REQUIRE j.title IS UNIQUE")
    for label in entity_labels:
        tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{label}) REQUIRE e.text IS UNIQUE")

# ------------------------------------------
# Step 3: Load resumes from SQLite
# ------------------------------------------

def load_resumes_from_sqlite():
    conn = sqlite3.connect(db_path)
    query = """
    SELECT r.id AS resume_id, r.resume_text
    FROM resumes r
    JOIN annotations a ON r.id = a.resume_id
    JOIN predicted_jobs pj ON r.id = pj.resume_id
    WHERE a.rating >= 3;
    """
    df_resumes = pd.read_sql_query(query, conn)
    conn.close()
    return df_resumes

def push_resume_node(tx, resume_id, resume_text):
    tx.run("""
        MERGE (r:Resume {id: $resume_id})
        SET r.resume_text = $resume_text
    """, resume_id=resume_id, resume_text=resume_text)

# ------------------------------------------
# Step 4: Push similarity records
# ------------------------------------------

def push_to_neo4j(tx, record, entity_label, entity_key):
    query = f"""
        MATCH (r:Resume {{id: $resume_id}})
        MERGE (n:NounPhrase {{text: $noun_phrase}})
        MERGE (e:{entity_label} {{text: $entity_text}})
        MERGE (j:JobTitle {{title: $job_title}})
        MERGE (r)-[:CONTAINS]->(n)
        MERGE (n)-[s:SIMILAR_TO]->(e)
        SET s.score = $similarity_score
        MERGE (e)-[rj:REQUIRED_FOR]->(j)
        SET rj.importance = $data_value
    """
    tx.run(query,
           resume_id=record["resume_id"],
           job_title=record["job_title"],
           noun_phrase=record["noun_phrase"],
           entity_text=record[entity_key],
           similarity_score=round(record["similarity_score"], 4),
           data_value=record["data_value"])

# ------------------------------------------
# Step 5: Main logic
# ------------------------------------------

def main():
    # Clear the existing graph
    clear_database()

    # Collect actual entity labels from filenames
    entity_labels = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv") and file.startswith("resume_"):
            category_name = file.replace("resume_", "").replace("_similarity_matrix.csv", "")
            entity_labels.append(category_name.capitalize())

    # Create schema constraints
    with driver.session() as session:
        session.execute_write(create_constraints, entity_labels)
        print(f"✅ Constraints created for: {entity_labels}")

    # Create Resume nodes from SQLite
    df_resumes = load_resumes_from_sqlite()
    with driver.session() as session:
        for _, row in df_resumes.iterrows():
            session.execute_write(push_resume_node, row["resume_id"], row["resume_text"])
    print(f"✅ Created {len(df_resumes)} Resume nodes from SQLite.")

    # Push similarity matrix data
    with driver.session() as session:
        for file in os.listdir(input_dir):
            if file.endswith(".csv") and file.startswith("resume_"):
                category_name = file.replace("resume_", "").replace("_similarity_matrix.csv", "")
                entity_label = category_name.capitalize()
                entity_key = f"{category_name}_entity"
                file_path = os.path.join(input_dir, file)

                logging.info(f"📂 Loading: {file_path}")
                df = pd.read_csv(file_path)

                if entity_key not in df.columns:
                    logging.warning(f"⚠️ Column '{entity_key}' not found in {file}")
                    continue

                for _, row in df.iterrows():
                    session.execute_write(push_to_neo4j, row, entity_label, entity_key)

                logging.info(f"✅ Finished pushing {category_name} data to Neo4j.")

    driver.close()
    logging.info("🎉 All data successfully pushed to Neo4j.")

# ------------------------------------------
# Entry point
# ------------------------------------------

if __name__ == "__main__":
    main()
