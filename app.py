import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="Property Match Engine", layout="wide")
st.title("🏠 Property Recommendation System")

# =====================================================
# Load SentenceTransformer model
# =====================================================
@st.cache_resource
def load_model():
    with open("match_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =====================================================
# Load property data
# =====================================================
@st.cache_data
def load_data():
    return pd.read_excel("properties.xlsx")

properties_df = load_data()

# Preserve original price for display
properties_df["Price_raw"] = properties_df["Price"]

# =====================================================
# Price cleaning for COMPUTATION ONLY
# $500K → 500000
# =====================================================
def clean_price_numeric(x):
    if pd.isna(x):
        return 0

    if isinstance(x, (int, float)):
        return int(x)

    x = str(x).strip().lower()
    x = x.replace("$", "").replace(",", "")

    if x.endswith("k"):
        return int(float(x[:-1]) * 1000)

    return int(float(x))


properties_df["Price_num"] = properties_df["Price"].apply(clean_price_numeric)

properties_df["Bedrooms"] = (
    pd.to_numeric(properties_df["Bedrooms"], errors="coerce")
    .fillna(0)
    .astype(int)
)

properties_df["Bathrooms"] = (
    pd.to_numeric(properties_df["Bathrooms"], errors="coerce")
    .fillna(0)
    .astype(int)
)

properties_df["Area"] = (
    pd.to_numeric(properties_df["Area"], errors="coerce")
    .fillna(0)
    .astype(int)
)

# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header("🔍 Your Preferences")

# Budget slider in $k
budget_k = st.sidebar.slider(
    "Budget ($k)",
    min_value=250,
    max_value=1000,
    value=500,
    step=50
)

# Convert to dollars for computation
budget = budget_k * 1000

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=5,
    step=1,
    value=1
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=4,
    step=1,
    value=1
)

description = st.sidebar.text_area(
    "Qualitative Description",
    placeholder="e.g. modern, spacious, close to metro, premium interiors"
)

# =====================================================
# Matching Logic
# =====================================================
def compute_match_scores(df, user_input, model):

    property_texts = df["Qualitative Description"].fillna("").tolist()
    user_text = user_input["description"]

    property_embeddings = model.encode(property_texts, show_progress_bar=False)
    user_embedding = model.encode([user_text], show_progress_bar=False)

    text_similarities = cosine_similarity(
        user_embedding, property_embeddings
    )[0]

    scores = []

    for i, row in df.iterrows():

        # Budget score
        budget_diff = abs(row["Price_num"] - user_input["budget"])
        budget_score = np.exp(
            -budget_diff / max(user_input["budget"], 1)
        )

        # Bedroom score
        bed_score = 1 - abs(
            row["Bedrooms"] - user_input["bedrooms"]
        ) / max(row["Bedrooms"], user_input["bedrooms"], 1)

        # Bathroom score
        bath_score = 1 - abs(
            row["Bathrooms"] - user_input["bathrooms"]
        ) / max(row["Bathrooms"], user_input["bathrooms"], 1)

        # Text similarity score
        text_score = text_similarities[i]

        final_score = (
            0.35 * budget_score +
            0.25 * bed_score +
            0.20 * bath_score +
            0.20 * text_score
        )

        scores.append(final_score)

    return scores

# =====================================================
# Run Recommendation
# =====================================================
if st.sidebar.button("Find Best Matches 🚀"):

    user_input = {
        "budget": budget,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "description": description
    }

    properties_df["Match Score"] = compute_match_scores(
        properties_df, user_input, model
    )

    top_5 = (
        properties_df
        .sort_values("Match Score", ascending=False)
        .head(5)
    )

    st.subheader("🏆 Top 5 Recommended Properties")

    st.dataframe(
        top_5[
            [
                "Property ID",
                "Price_raw",   # 👈 DISPLAY AS-IS FROM EXCEL
                "Bedrooms",
                "Bathrooms",
                "Area",
                "Qualitative Description",
                "Match Score"
            ]
        ].rename(columns={"Price_raw": "Price"})
        .style.format({"Match Score": "{:.3f}"})
    )

else:
    st.info("👈 Select preferences and click **Find Best Matches**")
