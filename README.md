# Shear,Shuffle and Predict
Data challenge: Predict future local dynamics (T1 events, D2min) in a 2D amorphous system from initial static configurations.

# Predicting Amorphous Dynamics Challenge

This repository contains data and a research-oriented machine learning challenge:  
**Can we predict the future dynamics of particles in a 2D amorphous system using only static structural information?**

The data is extracted from particle-based simulations of a confined amorphous material (e.g., a glassy system with an obstacle), and includes local dynamical markers like **T1 rearrangements** and **D2<sub>min</sub>** non-affine displacements.

---

## 🔍 Challenge Overview

You are given several `.txt` data files representing the **initial configurations** of particles, with associated:
- **T1 event flags** (1 if the particle undergoes a topological rearrangement)
- **D2<sub>min</sub> values** (a measure of local non-affine deformation)

Each file contains **500 configurations**. Your task is to build a model that **predicts the future T1 or D2<sub>min</sub> values** based on the current configuration.

---

## 🧪 Data Format

Each file (e.g., `t30.txt`, `t60.txt`, etc.) corresponds to a time window `t`, indicating how far into the future the dynamics were computed.

### Columns:
1. `x` – x-coordinate of the particle  
2. `y` – y-coordinate  
3. `d` – particle diameter  
4. `T1` – flag: 0 (no rearrangement), 1 (rearranged)  
5. `D2min` – scalar value indicating non-affine displacement

### Structure:
- Each file contains **500 configurations**
- Each configuration has **900 particles**
- Each configuration is separated by blocks of **900 rows**
- A header is included with the number of particles (`900`) and box dimensions

### Files:
- `t30.txt` → target computed at time window = 30  
- `t60.txt`  
- `t90.txt`  
- `t150.txt`  
- `t180.txt`

---

## 🎯 Goal

Use static information (`x`, `y`, `d`) to predict **future T1 or D2min** values for each particle.

This could involve:
- Classifying particles as likely to rearrange (binary classification)
- Regressing the D2<sub>min</sub> field (regression)
- Exploring which features or local structures are predictive of dynamics

---

## 📂 Folder Structure

```plaintext
/
├── data/
│   ├── t30.txt
│   ├── t60.txt
│   ├── t90.txt
│   ├── t150.txt
│   └── t180.txt
├── README.md
├── analysis_scripts/
│   ├── compute_d2min.py
│   ├── detect_t1_events.py
│   └── generate_data.py
