# Routine-Model-Discovery-by-Extracting-Routine-Logs-in-RPA


This repository presents a methodology to extract routine patterns from unsegmented UI logs. The project utilizes a clustering-based approach to group similar activity segments and reconstruct routine logs. Various clustering techniques including **KMeans**, **DBSCAN**, and **HDBSCAN** were used to identify latent routine structures. The quality of the extracted routines was evaluated using **Jaccard Similarity** and **Fitness Scores** against ground truth models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Step 1: Log Preparation and Unsegmentation](#step-1-log-preparation-and-unsegmentation)
  - [Step 2: Segmentation](#step-2-segmentation)
  - [Step 3: Activity Vector Encoding](#step-3-activity-vector-encoding)
  - [Step 4: Clustering](#step-4-clustering)
  - [Step 5: Routine Pattern Extraction](#step-5-routine-pattern-extraction)
  - [Step 6: Evaluation](#step-6-evaluation)
- [Example](#example)
- [Dependencies](#dependencies)
- [Usage](#usage)



---

## 🧠 Overview

In scenarios where UI activity logs are not segmented by user sessions or routines, it becomes challenging to discover patterns directly. This project addresses the problem by:

- Assuming all routines are performed by a **single user**
- Segmenting logs using **end-activity heuristics**
- Encoding segments as binary activity vectors
- Clustering encoded vectors to detect **routine patterns**
- Reconstructing and evaluating routine logs

---

## ⚙️ Methodology

### Step 1: Log Preparation and Unsegmentation

All activity logs were treated as if performed by a single user by assigning a common user ID. This was done to simulate cases where user-level data is either anonymized or unavailable.

### Step 2: Segmentation

The unsegmented log is split into segments based on **end-activity markers**. Each segment is assumed to represent an instance of a routine, ending at a pre-defined terminal activity (e.g., `activity_6`).

### Step 3: Activity Vector Encoding

Each segment is encoded into a **binary activity vector**, where:

- Each index in the vector corresponds to a unique activity.
- The value `1` denotes the presence of the activity in the segment.

#### Example

Consider two segments:
- Segment 1 = `{activity_1, activity_3, activity_4}`
- Segment 2 = `{activity_1, activity_2, activity_5}`

Let the activity space be `{activity_1, activity_2, activity_3, activity_4, activity_5}`  
The encoded vectors are:

Segment 1 → [1, 0, 1, 1, 0]
Segment 2 → [1, 1, 0, 0, 1]


### Step 4: Clustering

These encoded vectors are clustered using the following unsupervised algorithms:

- **KMeans** (specify number of clusters `k`)
- **DBSCAN** (density-based clustering)
- **HDBSCAN** (hierarchical density-based clustering)

Segments with similar activity compositions are grouped into the same cluster, revealing routine similarity.

### Step 5: Routine Pattern Extraction

From each cluster:

- A **representative pattern** (e.g., cluster centroid or frequent pattern) is extracted.
- Each segment is assigned to its corresponding cluster routine.
- Final **routine logs** are reconstructed based on the cluster assignments.

### Step 6: Evaluation

Each reconstructed routine log is compared against the **ground truth routine models** using:

- **Jaccard Similarity Score** – measuring set overlap between predicted and actual activity sets.
- **Fitness Score** – evaluating the conformance of the routine log with the actual routine model.

---

## 🧪 Example

Given:
- **Activity Space**: `{activity_1, activity_2, activity_3, activity_4, activity_5}`
- **Encoded Vectors**:

S1 → [1, 0, 1, 1, 0]
S2 → [1, 1, 0, 0, 1]
S3 → [1, 0, 1, 1, 0]
S4 → [1, 1, 0, 0, 1]

Using KMeans (`k=2`), we may obtain:

- **Cluster 0**: S1, S3 (pattern: [1, 0, 1, 1, 0])
- **Cluster 1**: S2, S4 (pattern: [1, 1, 0, 0, 1])

Resulting routine logs:

- **Routine A** → `{activity_1, activity_3, activity_4}`
- **Routine B** → `{activity_1, activity_2, activity_5}`

---

## 🧰 Dependencies

- `Python 3.8+`
- `scikit-learn`
- `hdbscan`
- `pandas`
- `numpy`
- `matplotlib` *(optional, for visualization)*

# Usage
git clone https://github.com/yourusername/Routine-Model-Discovery-by-Extracting-Routine-Logs-in-RPA.git
cd Routine-Model-Discovery-by-Extracting-Routine-Logs-in-RPA

Start Jupyter here and run the following files:
- 01 RoutineDiscovery_and_Evaluate (Our Technique)
- 02 RoutineDiscovery_and_Evaluate (Our Technique) - Nooisy Logs