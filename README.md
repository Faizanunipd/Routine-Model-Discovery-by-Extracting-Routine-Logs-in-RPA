# Routine-Model-Discovery-by-Extracting-Routine-Logs-in-RPA


This repository presents a methodology to extract routine logs from unsegmented UI logs. The project utilizes a clustering-based approach to group similar segments and reconstruct routine logs. Various clustering techniques including **KMeans**, **DBSCAN**, and **HDBSCAN** were used to identify routine logs. The quality of the extracted routine logs was evaluated using **Jaccard Coefficient** and **Fitness Scores** against ground truth models.

---

## 📋 Table of Contents

- [Overview](#overview)
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
- 02 RoutineDiscovery_and_Evaluate (Our Technique) - Noisy Logs
