# 📥 Load and Prepare O*NET Technology and Tools Data

```python
import pandas as pd
import os

data_dir = "../data/o_net_tools_tech/"
# Load from Excel

tools_file = "Tools Used.xlsx"
tech_file = "Technology Skills.xlsx"

# Load Excel files
tech_df = pd.read_excel(os.path.join(data_dir, tech_file))
tools_df = pd.read_excel(os.path.join(data_dir, tools_file))

# Add identifier column
tech_df['Entity_Type'] = 'Technology'
tools_df['Entity_Type'] = 'Tool'

```


```python
# Check dataset summaries and structure
print ("tech df:", tech_df.info())
print ("tools df:", tools_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32627 entries, 0 to 32626
    Data columns (total 8 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   O*NET-SOC Code   32627 non-null  object
     1   Title            32627 non-null  object
     2   Example          32627 non-null  object
     3   Commodity Code   32627 non-null  int64 
     4   Commodity Title  32627 non-null  object
     5   Hot Technology   32627 non-null  object
     6   In Demand        32627 non-null  object
     7   Entity_Type      32627 non-null  object
    dtypes: int64(1), object(7)
    memory usage: 2.0+ MB
    tech df: None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41662 entries, 0 to 41661
    Data columns (total 6 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   O*NET-SOC Code   41662 non-null  object
     1   Title            41662 non-null  object
     2   Example          41662 non-null  object
     3   Commodity Code   41662 non-null  int64 
     4   Commodity Title  41662 non-null  object
     5   Entity_Type      41662 non-null  object
    dtypes: int64(1), object(5)
    memory usage: 1.9+ MB
    tools df: None



```python
tech_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>O*NET-SOC Code</th>
      <th>Title</th>
      <th>Example</th>
      <th>Commodity Code</th>
      <th>Commodity Title</th>
      <th>Hot Technology</th>
      <th>In Demand</th>
      <th>Entity_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>Adobe Acrobat</td>
      <td>43232202</td>
      <td>Document management software</td>
      <td>Y</td>
      <td>N</td>
      <td>Technology</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>AdSense Tracker</td>
      <td>43232306</td>
      <td>Data base user interface and query software</td>
      <td>N</td>
      <td>N</td>
      <td>Technology</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>Atlassian JIRA</td>
      <td>43232201</td>
      <td>Content workflow software</td>
      <td>Y</td>
      <td>N</td>
      <td>Technology</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>Blackbaud The Raiser's Edge</td>
      <td>43232303</td>
      <td>Customer relationship management CRM software</td>
      <td>N</td>
      <td>N</td>
      <td>Technology</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>ComputerEase construction accounting software</td>
      <td>43231601</td>
      <td>Accounting software</td>
      <td>N</td>
      <td>N</td>
      <td>Technology</td>
    </tr>
  </tbody>
</table>
</div>




# 📄 Load and Filter Resume Data

```python
# lets pick a resume
resume_df = pd.read_csv("../data/annotations_scenario_1/cleaned_resumes.csv")
```


```python

# Select resume labeled for 'data science' role
data_science_resume = resume_df[resume_df['original_job'].str.lower() == "data science"]

# Display the result
# Extract the first matching resume text
first_ds_resume = data_science_resume.iloc[0]

# Extract the resume text
resume_text = first_ds_resume['resume_text']

# Show a short preview of the resume for verification
print(resume_text[:1000])  # show first 1000 characters


```

    Education Details B. Tech Rayat and Bahra Institute of Engineering and Biotechnology Data Science Data Science Skill Details Numpy- Exprience - Less than 1 year months Machine Learning- Exprience - Less than 1 year months Tensorflow- Exprience - Less than 1 year months Scikit- Exprience - Less than 1 year months Python- Exprience - Less than 1 year months GCP- Exprience - Less than 1 year months Pandas- Exprience - Less than 1 year months Neural Network- Exprience - Less than 1 year monthsCompany Details company - Wipro description - Bhawana Aggarwal E-Mail:bhawana. chd@gmail. com Phone: 09876971076 VVersatile, high-energy professional targeting challenging assignments in Machine PROFILE SUMMARY An IT professional with knowledge and experience of 2 years in Wipro Technologies in Machine Learning, Deep Learning, Data Science, Python, Software Development. Skilled in managing end-to-end development and software products / projects from inception, requirement specs, planning, designing, i



```python
# Check CUDA availability
import torch
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model to CUDA
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
```

    Using device: cuda



```python
from textblob import TextBlob

blob = TextBlob(resume_text)

# Inspecting basic information
print(f"Text length: {len(blob.words)} words")
print(f"Sentences count: {len(blob.sentences)}")
```

    Text length: 1120 words
    Sentences count: 59



```python
noun_phrases = blob.noun_phrases
print(f"Noun phrases found ({len(noun_phrases)}):\n")
for np in noun_phrases[:20]:
    print("-", np)
```

    Noun phrases found (335):
    
    - details
    - tech rayat
    - bahra
    - engineering
    - biotechnology data
    - data
    - skill details numpy- exprience
    - less
    - year months
    - machine learning- exprience
    - less
    - year months
    - tensorflow- exprience
    - less
    - year months
    - scikit- exprience
    - less
    - year months
    - python- exprience
    - less



```python
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

# -------------------------------
# 1. Load Model on GPU
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# -------------------------------
# 2. Load Tools and Tech Data
# -------------------------------
tools_df = pd.read_excel("../data/o_net_tools_tech/Tools Used.xlsx")
tech_df = pd.read_excel("../data/o_net_tools_tech/Technology Skills.xlsx")

# -------------------------------
# 3. Clean Text Columns
# -------------------------------
tools_df = tools_df.dropna(subset=['Example'])
tools_df['tool_example'] = tools_df['Example'].astype(str)

tech_df = tech_df.dropna(subset=['Example'])
tech_df['tech_example'] = tech_df['Example'].astype(str)

# -------------------------------
# 4. Batch Encode Tool Examples
# -------------------------------
tool_texts = tools_df['tool_example'].tolist()

print("Encoding Tools...")
tool_embeddings = model.encode(
    tool_texts,
    batch_size=64,                    # Adjust based on GPU memory
    convert_to_numpy=True,
    show_progress_bar=True
)
tools_df['embedding'] = list(tool_embeddings)

# -------------------------------
# 5. Batch Encode Tech Examples
# -------------------------------
tech_texts = tech_df['tech_example'].tolist()

print("Encoding Technology Skills...")
tech_embeddings = model.encode(
    tech_texts,
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True
)
tech_df['embedding'] = list(tech_embeddings)

# -------------------------------
# 6. Final Check
# -------------------------------
print(f"Encoded {len(tools_df)} tools and {len(tech_df)} technology skills.")

```

    Using device: cuda
    Encoding Tools...



    Batches:   0%|          | 0/651 [00:00<?, ?it/s]


    Encoding Technology Skills...



    Batches:   0%|          | 0/510 [00:00<?, ?it/s]


    Encoded 41662 tools and 32627 technology skills.



```python
tech_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>O*NET-SOC Code</th>
      <th>Title</th>
      <th>Example</th>
      <th>Commodity Code</th>
      <th>Commodity Title</th>
      <th>Hot Technology</th>
      <th>In Demand</th>
      <th>tech_example</th>
      <th>embedding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>Adobe Acrobat</td>
      <td>43232202</td>
      <td>Document management software</td>
      <td>Y</td>
      <td>N</td>
      <td>Adobe Acrobat</td>
      <td>[-0.09180075, 0.0021334619, -0.13546862, 0.008...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>AdSense Tracker</td>
      <td>43232306</td>
      <td>Data base user interface and query software</td>
      <td>N</td>
      <td>N</td>
      <td>AdSense Tracker</td>
      <td>[-0.070951834, -0.03741923, -0.045952767, 0.04...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>Atlassian JIRA</td>
      <td>43232201</td>
      <td>Content workflow software</td>
      <td>Y</td>
      <td>N</td>
      <td>Atlassian JIRA</td>
      <td>[-0.1495522, -0.015270967, -0.039499376, -0.03...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>Blackbaud The Raiser's Edge</td>
      <td>43232303</td>
      <td>Customer relationship management CRM software</td>
      <td>N</td>
      <td>N</td>
      <td>Blackbaud The Raiser's Edge</td>
      <td>[-0.1006839, 0.03715423, -0.096757166, 0.00728...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11-1011.00</td>
      <td>Chief Executives</td>
      <td>ComputerEase construction accounting software</td>
      <td>43231601</td>
      <td>Accounting software</td>
      <td>N</td>
      <td>N</td>
      <td>ComputerEase construction accounting software</td>
      <td>[-0.07286814, 0.07547406, -0.0423134, -0.01320...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename for Tools
if 'O*NET-SOC Code' in tools_df.columns:
    tools_df = tools_df.rename(columns={'O*NET-SOC Code': 'onetsoc_code'})

# Rename for Technology
if 'O*NET-SOC Code' in tech_df.columns:
    tech_df = tech_df.rename(columns={'O*NET-SOC Code': 'onetsoc_code'})

```


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# --------------------------------------------
# Step 1: Prepare Resume Embedding Matrix
# --------------------------------------------
# Stack noun phrase embeddings into a NumPy array
np_embeddings = np.vstack(nounphrase_df['embedding'].values)

# --------------------------------------------
# Step 2: Match Tools Used to Resume
# --------------------------------------------
tool_matches = []

for idx, row in tools_df.rename(columns={'O*NET-SOC Code': 'onetsoc_code'}).iterrows():
    tool_name = str(row['tool_example'])
    tool_embedding = row['embedding']
    job_code = row['onetsoc_code']

    # Compute cosine similarity with all noun phrases
    similarities = cosine_similarity([tool_embedding], np_embeddings)[0]
    max_score = similarities.max()

    if max_score >= 0.65:
        matched_idx = similarities.argmax()
        tool_matches.append({
            'onetsoc_code': job_code,
            'tool_example': tool_name,
            'matched_noun': nounphrase_df.iloc[matched_idx]['noun_phrase'],
            'similarity': max_score,
            'tool_score': 1.0  # Per paper: binary match
        })

tools_matched_df = pd.DataFrame(tool_matches)

# --------------------------------------------
# Step 3: Match Technology Skills to Resume
# --------------------------------------------
tech_matches = []

for idx, row in tech_df.rename(columns={'O*NET-SOC Code': 'onetsoc_code'}).iterrows():
    tech_name = str(row['tech_example'])
    tech_embedding = row['embedding']
    job_code = row['onetsoc_code']
    is_hot = row.get('hot_technology', 'N')

    # Compute cosine similarity with all noun phrases
    similarities = cosine_similarity([tech_embedding], np_embeddings)[0]
    max_score = similarities.max()

    if max_score >= 0.65:
        matched_idx = similarities.argmax()
        tech_score = 1.0 if is_hot == 'Y' else 0.75
        tech_matches.append({
            'onetsoc_code': job_code,
            'tech_example': tech_name,
            'matched_noun': nounphrase_df.iloc[matched_idx]['noun_phrase'],
            'similarity': max_score,
            'tech_score': tech_score
        })

tech_matched_df = pd.DataFrame(tech_matches)

# --------------------------------------------
# Step 4: Aggregate Scores by Job
# --------------------------------------------
tool_scores_by_job = tools_matched_df.groupby('onetsoc_code')['tool_score'].sum().reset_index()
tech_scores_by_job = tech_matched_df.groupby('onetsoc_code')['tech_score'].sum().reset_index()

# Merge both scores by job code
job_tech_tool_scores = pd.merge(tool_scores_by_job, tech_scores_by_job, on='onetsoc_code', how='outer').fillna(0)

# Compute total score (tool + tech)
job_tech_tool_scores['total_score'] = job_tech_tool_scores['tool_score'] + job_tech_tool_scores['tech_score']

# --------------------------------------------
# Step 5: Show Top Matching Jobs
# --------------------------------------------
job_tech_tool_scores = job_tech_tool_scores.sort_values(by='total_score', ascending=False)
display(job_tech_tool_scores.head(10))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>onetsoc_code</th>
      <th>tool_score</th>
      <th>tech_score</th>
      <th>total_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>117</th>
      <td>15-1252.00</td>
      <td>2.0</td>
      <td>23.25</td>
      <td>25.25</td>
    </tr>
    <tr>
      <th>118</th>
      <td>15-1253.00</td>
      <td>5.0</td>
      <td>18.00</td>
      <td>23.00</td>
    </tr>
    <tr>
      <th>113</th>
      <td>15-1243.00</td>
      <td>3.0</td>
      <td>18.75</td>
      <td>21.75</td>
    </tr>
    <tr>
      <th>112</th>
      <td>15-1242.00</td>
      <td>2.0</td>
      <td>19.50</td>
      <td>21.50</td>
    </tr>
    <tr>
      <th>104</th>
      <td>15-1211.00</td>
      <td>2.0</td>
      <td>18.00</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>129</th>
      <td>15-1299.08</td>
      <td>4.0</td>
      <td>15.75</td>
      <td>19.75</td>
    </tr>
    <tr>
      <th>130</th>
      <td>15-1299.09</td>
      <td>4.0</td>
      <td>13.50</td>
      <td>17.50</td>
    </tr>
    <tr>
      <th>77</th>
      <td>13-1111.00</td>
      <td>3.0</td>
      <td>14.25</td>
      <td>17.25</td>
    </tr>
    <tr>
      <th>110</th>
      <td>15-1241.00</td>
      <td>4.0</td>
      <td>12.75</td>
      <td>16.75</td>
    </tr>
    <tr>
      <th>120</th>
      <td>15-1255.00</td>
      <td>3.0</td>
      <td>13.50</td>
      <td>16.50</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Total required tools/tech per job (from O*NET, before matching)
total_tools_by_job = tools_df.groupby('onetsoc_code')['tool_example'].nunique().reset_index(name='total_tools')
total_tech_by_job = tech_df.groupby('onetsoc_code')['tech_example'].nunique().reset_index(name='total_tech')
matched_tools_count = tools_matched_df.groupby('onetsoc_code')['tool_example'].nunique().reset_index(name='matched_tools')
matched_tech_count = tech_matched_df.groupby('onetsoc_code')['tech_example'].nunique().reset_index(name='matched_tech')

```


```python
# Start with scores
analytics_df = job_tech_tool_scores.copy()

# Merge in matched counts
analytics_df = analytics_df.merge(matched_tools_count, on='onetsoc_code', how='left').fillna(0)
analytics_df = analytics_df.merge(matched_tech_count, on='onetsoc_code', how='left').fillna(0)

# Merge in totals for normalization
analytics_df = analytics_df.merge(total_tools_by_job, on='onetsoc_code', how='left').fillna(0)
analytics_df = analytics_df.merge(total_tech_by_job, on='onetsoc_code', how='left').fillna(0)

# Calculate percentage match
analytics_df['tool_coverage_pct'] = (analytics_df['matched_tools'] / analytics_df['total_tools']).replace([np.inf, np.nan], 0)
analytics_df['tech_coverage_pct'] = (analytics_df['matched_tech'] / analytics_df['total_tech']).replace([np.inf, np.nan], 0)

```


```python
import matplotlib.pyplot as plt

# Top 10 jobs by total score
top_jobs = analytics_df.sort_values(by='total_score', ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.barh(top_jobs['onetsoc_code'], top_jobs['total_score'], color='skyblue')
plt.xlabel("Total Score (Tools + Tech)")
plt.ylabel("O*NET-SOC Code")
plt.title("Top 10 Jobs Matched by Tools and Technology Skills")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

```


    
![png](output_14_0.png)
    



```python
# Add a 'source' tag to differentiate
tools_matched_df['source'] = 'Tool'
tech_matched_df['source'] = 'Tech'

# Standardize column names for merge
tools_matched_df = tools_matched_df.rename(columns={
    'tool_example': 'entity_example',
    'tool_score': 'score'
})

tech_matched_df = tech_matched_df.rename(columns={
    'tech_example': 'entity_example',
    'tech_score': 'score'
})

# Combine
combined_matches_df = pd.concat([tools_matched_df, tech_matched_df], ignore_index=True)

```


```python
# Group by matched noun
top_hit_terms = combined_matches_df.groupby('matched_noun').agg(
    match_count=('onetsoc_code', 'count'),
    total_score=('score', 'sum'),
    top_examples=('entity_example', lambda x: list(x.unique())[:3])  # sample top matches
).reset_index()

# Sort by total score or match count
top_hit_terms = top_hit_terms.sort_values(by='total_score', ascending=False)

# Show top 10 resume phrases that had highest overall impact
display(top_hit_terms.head(10))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>matched_noun</th>
      <th>match_count</th>
      <th>total_score</th>
      <th>top_examples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>computer</td>
      <td>1651</td>
      <td>1651.00</td>
      <td>[Desktop computers, Laptop computers, Personal...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>software</td>
      <td>1062</td>
      <td>796.50</td>
      <td>[Microsoft Office software, Salesforce softwar...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>oracle</td>
      <td>517</td>
      <td>387.75</td>
      <td>[Oracle PeopleSoft, Oracle Database, Oracle El...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>windows</td>
      <td>274</td>
      <td>205.50</td>
      <td>[Microsoft Windows, Microsoft Windows XP, Micr...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>linux</td>
      <td>210</td>
      <td>157.50</td>
      <td>[Linux, UNIX]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>notebooks</td>
      <td>146</td>
      <td>146.00</td>
      <td>[Notebook computers, Dell Notebooks]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>python</td>
      <td>131</td>
      <td>98.25</td>
      <td>[Python]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c++</td>
      <td>110</td>
      <td>82.50</td>
      <td>[C++]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>phone</td>
      <td>41</td>
      <td>41.00</td>
      <td>[Smartphones, Cell phones, Mobile phones]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>mysql</td>
      <td>50</td>
      <td>37.50</td>
      <td>[MySQL]</td>
    </tr>
  </tbody>
</table>
</div>



```python
import matplotlib.pyplot as plt

top_n = 10
plt.figure(figsize=(12,6))
plt.barh(top_hit_terms['matched_noun'].head(top_n)[::-1], top_hit_terms['total_score'].head(top_n)[::-1])
plt.xlabel("Cumulative Score (Tools + Tech)")
plt.title("Top Resume Terms with Highest O*NET Tool/Tech Match Impact")
plt.tight_layout()
plt.show()

```


    
![png](output_17_0.png)
    

