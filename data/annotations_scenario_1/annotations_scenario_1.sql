DROP TABLE IF EXISTS resumes;
DROP TABLE IF EXISTS predicted_jobs;
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS naive_algorithm_predictions;
DROP TABLE IF EXISTS naive_algorithm_annotations;

CREATE TABLE resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_text TEXT NOT NULL,
    original_job TEXT NOT NULL
);

CREATE TABLE predicted_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    job_code TEXT NOT NULL,
    job_title TEXT NOT NULL,
    FOREIGN KEY (resume_id) REFERENCES resumes(id)
);

CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    annotator_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    FOREIGN KEY (resume_id) REFERENCES resumes(id)
);

CREATE TABLE naive_algorithm_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    algorithm_version TEXT NOT NULL,
    job_title TEXT NOT NULL,
    similarity_score FLOAT NOT NULL,
    FOREIGN KEY (resume_id) REFERENCES resumes(id)
);

CREATE TABLE naive_algorithm_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    algorithm_version TEXT NOT NULL,
    annotator_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    FOREIGN KEY (resume_id) REFERENCES resumes(id)
);
