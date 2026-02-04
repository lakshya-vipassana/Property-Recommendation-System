import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="Property Match Engine", layout="wide")
st.title("🏠 Property Recommendation System")

# =====================================================
# Load model
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("model/", device="cpu")

model = load_model()   # ✅ IMPORTANT

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
# Price cleaning (computation only)
# =====================================================
def clean_price_numeric(x):
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float)):
        return int(x)

    x = str(x).lower().replace("$", "").replace(",", "")
    if x.endswith("k"):
        return int(float(x[:-1]) * 1000)
    return int(float(x))

properties_df["Price_num"] = properties_df["Price"].apply(clean_price_numeric)

properties_df["Bedrooms"] = pd.to_numeric(
    properties_df["Bedrooms"], errors="coerce"
).fillna(0).astype(int)

properties_df["Bathrooms"] = pd.to_numeric(
    properties_df["Bathrooms"], errors="coerce"
).fillna(0).astype(int)

properties_df["Area"] = pd.to_numeric(
    properties_df["Area"], errors="coerce"
).fillna(0).astype(int)

# =====================================================
# Precompute property embeddings (VERY IMPORTANT)
# =====================================================
@st.cache_resource
def compute_property_embeddings(df, model):
    texts = df["Qualitative Description"].fillna("").tolist()
    return model.encode(texts, show_progress_bar=False)

property_embeddings = compute_property_embeddings(properties_df, model)

# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header("🔍 Your Preferences")

budget_k = st.sidebar.slider(
    "Budget ($k)", 250, 1000, 500, 50
)
budget = budget_k * 1000

bedrooms = st.sidebar.number_input(
    "Bedrooms", min_value=1, max_value=5, value=1
)

bathrooms = st.sidebar.number_input(
    "Bathrooms", min_value=1, max_value=4, value=1
)

description = st.sidebar.text_area(
    "Qualitative Description",
    placeholder="e.g. modern, spacious, close to metro"
)

# =====================================================
# Matching Logic
# =====================================================
def compute_match_scores(df, user_input, model, property_embeddings):

    user_embedding = model.encode(
        [user_input["description"]],
        show_progress_bar=False
    )

    text_similarities = cosine_similarity(
        user_embedding, property_embeddings
    )[0]

    scores = []

    for i, row in df.iterrows():

        budget_diff = abs(row["Price_num"] - user_input["budget"])
        budget_score = np.exp(
            -budget_diff / max(user_input["budget"], 1)
        )

        bed_score = 1 - abs(
            row["Bedrooms"] - user_input["bedrooms"]
        ) / max(row["Bedrooms"], user_input["bedrooms"], 1)

        bath_score = 1 - abs(
            row["Bathrooms"] - user_input["bathrooms"]
        ) / max(row["Bathrooms"], user_input["bathrooms"], 1)

        final_score = (
            0.35 * budget_score +
            0.25 * bed_score +
            0.20 * bath_score +
            0.20 * text_similarities[i]
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
        properties_df, user_input, model, property_embeddings
    )

    top_5 = properties_df.sort_values(
        "Match Score", ascending=False
    ).head(5)

    st.subheader("🏆 Top 5 Recommended Properties")

    st.dataframe(
        top_5[
            [
                "Property ID",
                "Price_raw",
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
