# 🏠 Property Recommendation System

**What this system does:**  
A first-pass property matching and ranking system that aligns user preferences with real estate listings using hard feasibility filters and interpretable similarity-based scoring.

**Core logic vs demo:**  
The core system implements the matching and ranking methodology; the Streamlit app is a lightweight demo to showcase end-to-end usage and explainability.

**How to run:**  
Run locally via Streamlit or launch using the provided Docker (amd64) setup for presentation purposes.

---

An end-to-end **property matching and ranking system** that aligns user preferences with real estate listings using a combination of **hard constraints and feature-level similarity scoring**.  
The system explicitly separates **feasibility filtering** from **preference-based ranking**, ensuring both correctness and personalization.

Built using **Python, scikit-learn, sentence-transformers, and Streamlit**.

---

## 🚀 Key Features

- Preference-based property matching and ranking
- **Hard constraint filtering** (budget, minimum bedrooms)
- **Semantic intent matching** using sentence embeddings
- Rule-based amenity overlap scoring
- Weighted, interpretable match score
- Interactive web interface built with Streamlit
- Modular and extensible architecture
- Dockerized demo deployment (amd64)

---

## 🧠 System Architecture

User Preferences
↓
Hard Filters (Budget, Bedrooms)
↓
Feature-Level Similarity Computation
↓
Weighted Match Score Aggregation
↓
Top-N Ranked Property Recommendations


**Key Design Insight:**  
Feasibility is enforced first; preferences are ranked afterward.

---

## 🛠️ Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- sentence-transformers
- Streamlit
- joblib / pickle (for model persistence)

---

## ⚙️ How the System Works

1. The user provides preferences such as budget, minimum bedrooms, and a free-text description.
2. Properties that violate hard constraints (e.g., budget, bedrooms) are filtered out.
3. Remaining properties are represented using feature-level similarity signals:
   - Semantic text similarity
   - Amenity overlap
   - Price compatibility
   - Area similarity (weak contextual signal)
4. A weighted match score is computed for each feasible property.
5. Properties are ranked by score, and the top-N results are returned.

Weights are manually chosen for **interpretability**, not learned, due to the absence of labeled ground truth.

---

## 📊 Evaluation & Explainability

This is an **unsupervised ranking problem** with no explicit ground-truth labels.

Evaluation focuses on:
- Qualitative sanity checks
- Score separation across ranked results
- Feature contribution analysis using bar charts and heatmaps

The system emphasizes **explainability and alignment with user intent** rather than accuracy metrics.

---

## 🖥️ Demo & Deployment

A lightweight **Streamlit demo application** was deployed using **Docker (amd64)** on a cloud VM to demonstrate end-to-end usage.

⚠️ This deployment is a **presentation prototype**, not a production system.

Live Demo:  
👉 https://property-matching.azurewebsites.net/

---

## ▶️ Running the Application Locally

```bash
git clone https://github.com/lakshya-vipassana/property-recommendation-system.git
cd property-recommendation-system
pip install -r requirements.txt
streamlit run app.py
