<img src="logo/A_digital_vector_illustration_features_a_black_sil.png" alt="ReculseNet Logo" width="150"/>

# ReculseNet: PD Similarity and Redundancy Detection using Graph Analytics

ReculseNet is a project aimed at detecting redundancies in organizational Position Descriptions (PDs) and providing recommendations for workforce optimization. This is achieved using semantic similarity models, graph-based analysis, and Graph Neural Networks (GNNs). The project leverages the O*NET database for job skills and responsibility mapping.

## Data

O Net V29.0


## Project Structure

- **Data Preprocessing**: Noun phrase with textblob
- **Modeling**: embedding with sentence transformer
- **Graph Analytics**: Use Neo4j and Graph Data Science (GDS) to analyze job-role redundancies, skill overlaps, and job clusters.
- **Evaluation**: Validate results with human-annotated datasets and provide actionable recommendations.

## Installation

### Prerequisites

- Python 3.x
- Neo4j (if using Neo4j database)
- Docker (optional, if using Docker containers)
- sqlite3 (apt install)

### Set up the environment

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ReculseNet.git
    ```

2. Navigate to the project directory:
    ```bash
    cd ReculseNet
    ```

3. Create a Python virtual environment:
    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scriptsctivate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Run notebooks/SQLiteSetUP

## Usage

1. Preprocess PDs and resumes to generate graph nodes and edges.
3. Apply graph metrics (e.g., Louvain, PageRank) for redundancy detection and workforce optimization.
4. Train and evaluate the Graph Neural Network for PD clustering and predictive matching.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Pachidi et al. (2023) for their work on transformers and O*NET for job matching.
- Neo4j for graph database management and GDS tools.
