import os
import pandas as pd
import logging
from neo4j import GraphDatabase
import re

# -----------------------------
# Neo4j connection config
# -----------------------------
NEO4J_URI = "bolt://20.14.162.151:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "recluse2025"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------
# Load data
# -----------------------------
volcano_dir = "../data/volcano"
trait_csv = "Job_traits.csv"
trait_map_csv = "trait_to_ksas_mapping.csv"
occupation_raw_csv = "Occupations.csv"
occupation_match_csv = "occupation_to_jobtitle_mapping.csv"

traits_df = pd.read_csv(os.path.join(volcano_dir, trait_csv))
trait_map_df = pd.read_csv(os.path.join(volcano_dir, trait_map_csv))
occupation_raw_df = pd.read_csv(os.path.join(volcano_dir, occupation_raw_csv))
occupation_match_df = pd.read_csv(os.path.join(volcano_dir, occupation_match_csv))

# Merge: keep only occupations that matched a job title
occupation_df = occupation_match_df.merge(
    occupation_raw_df,
    left_on="occupation_name",
    right_on="Occupation",
    how="inner"
)

# -----------------------------
# Neo4j push functions
# -----------------------------

def sanitize_label(label):
    """Sanitize cluster names for Neo4j labels."""
    clean = re.sub(r'[^a-zA-Z0-9]', '_', label.strip())
    if not clean or not clean[0].isalpha():
        clean = "Cluster_" + clean
    return clean

def create_indexes(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Trait) REQUIRE t.text IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (t:Trait) ON (t.trait_cluster)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (t:Trait) ON (t.score1)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (t:Trait) ON (t.score2)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (t:Trait) ON (t.score3)")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Occupation) REQUIRE o.onetsoc_code IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (o:Occupation) ON (o.cluster)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (o:Occupation) ON (o.name)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (o:Occupation) ON (o.score1)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (o:Occupation) ON (o.score2)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (o:Occupation) ON (o.score3)")

def push_trait_node(tx, row):
    cluster_label = sanitize_label(row["Trait_Cluster"])

    query = f"""
        MERGE (t:Trait:{cluster_label} {{text: $text}})
        SET t.trait_cluster = $cluster,
            t.score1 = $score1,
            t.score2 = $score2,
            t.score3 = $score3
    """

    tx.run(query,
           text=row["Job_Trait"],
           cluster=row["Trait_Cluster"],
           score1=row["Score_Comp_.1"],
           score2=row["Score_Comp_.2"],
           score3=row["Score_Comp_.3"])

def push_trait_alignment(tx, row):
    tx.run("""
        MATCH (t:Trait {text: $trait})
        MATCH (e)
        WHERE toLower(e.text) = toLower($entity) AND $category IN labels(e)
        MERGE (t)-[:ALIGNS_WITH]->(e)
    """,
    trait=row["trait_text"],
    entity=row["matched_entity_text"],
    category=row["matched_category"].capitalize())

def push_occupation_node(tx, row):
    tx.run("""
        MERGE (o:Occupation {onetsoc_code: $code})
        SET o.name = $name,
            o.cluster = $cluster,
            o.score1 = $score1,
            o.score2 = $score2,
            o.score3 = $score3
    """,
    code=row["onetsoc_code"],
    name=row["occupation_name"],
    cluster=row["Occupation_Cluster"],
    score1=row["Score_Comp_.1"],
    score2=row["Score_Comp_.2"],
    score3=row["Score_Comp_.3"])

def align_occupation_to_jobtitle(tx, onetsoc_code):
    tx.run("""
        MATCH (o:Occupation {onetsoc_code: $code})
        MATCH (j:JobTitle {onetsoc_code: $code})
        MERGE (o)-[:ALIGNED_WITH]->(j)
    """, code=onetsoc_code)

# -----------------------------
# Main logic
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with driver.session() as session:
        logging.info("📐 Creating constraints and indexes...")
        session.execute_write(create_indexes)

        logging.info("📌 Pushing Trait nodes...")
        for _, row in traits_df.iterrows():
            session.execute_write(push_trait_node, row)

        logging.info("🔗 Linking Trait → KSA...")
        for _, row in trait_map_df.iterrows():
            session.execute_write(push_trait_alignment, row)

        logging.info("📌 Pushing Occupation nodes...")
        for _, row in occupation_df.iterrows():
            session.execute_write(push_occupation_node, row)

        logging.info("🔗 Linking Occupation → JobTitle...")
        for code in occupation_df["onetsoc_code"].unique():
            session.execute_write(align_occupation_to_jobtitle, code)

    driver.close()
    logging.info("✅ Trait and Occupation data pushed to Neo4j.")

# -----------------------------
# Run it
# -----------------------------
if __name__ == "__main__":
    main()
