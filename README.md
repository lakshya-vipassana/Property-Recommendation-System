# 🏠 Property Recommendation System

An end-to-end **property recommendation engine** that matches user preferences with real estate listings using **feature similarity and ranking logic**.  
The system separates **feasibility filtering** from **preference-based ranking**, ensuring both correctness and personalization.

Built using **Python, scikit-learn, and Streamlit**.

---

## 🚀 Features

- User preference–based property matching
- Hard filtering on critical constraints (budget, bedrooms, location, etc.)
- Similarity-based ranking using cosine similarity
- Weighted feature scoring for personalized recommendations
- Interactive web interface built with Streamlit
- Modular and extensible architecture

---

## 🧠 System Architecture

User Input
|
v
Hard Filters (Budget, Bedrooms, Location)
|
v
Feature Engineering & Normalization
|
v
Cosine Similarity Calculation
|
v
Weighted Match Score
|
v
Top-N Property Recommendations

**Key Design Insight:**  
Feasibility is handled first, preferences are ranked later.

---

## 🛠️ Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- Streamlit
- Pickle (for model persistence)
- sentence-transformers

---

## ⚙️ How the System Works

1. User provides preferences such as budget, bedrooms, and location.
2. Properties that violate hard constraints are filtered out.
3. Remaining properties are converted into numerical feature vectors.
4. Cosine similarity is computed between user preferences and properties.
5. Feature-wise weights are applied to compute a final match score.
6. Top-ranked properties are displayed to the user.

---

## ▶️ Running the Application Locally

Clone the repository: git clone https://github.com/lakshya-vipassana/property-recommendation-system.git

cd property-recommendation-system

Install dependencies: pip install -r requirements.txt

Run the Streamlit app: streamlit run app.py

---

## 📊 Output

- Ranked list of properties
- Match scores indicating relevance
- Real-time interactive recommendations

---

## 🎯 Use Cases

- Real estate recommendation platforms
- Personalized property search
- Data science and machine learning case studies
- Decision-support systems

---

## 📌 Future Enhancements

- Geospatial intelligence for location-based scoring
- NLP-based preference extraction
- Deep learning recommendation models
- Cloud deployment and CI/CD integration
- User feedback loop for adaptive learning

---

## 👤 Author

**Lakshya Vipassana**   
Indian Institute of Technology Kharagpur

---

## ⭐ Acknowledgements

- scikit-learn documentation
- Streamlit community
- Open-source Python ecosystem

If you find this project useful, consider starring the repository.
