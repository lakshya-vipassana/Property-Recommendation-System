# ===============================
# Core libraries
# ===============================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ===============================
# Load data from Excel
# ===============================
user_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="User Data")
property_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="Property Data")

# Quick sanity checks
user_df.head()
property_df.head()
user_df.info()
property_df.info()

# ===============================
# Utility: parse budget / price strings like "$500k"
# ===============================
def parse_money(x):
    x = x.replace('$','').strip()
    if 'k' in x.lower():
        return float(x.lower().replace('k','')) * 1000
    return float(x)

# Convert budget & price to numeric
user_df['Budget'] = user_df['Budget'].apply(parse_money)
property_df['Price'] = property_df['Price'].apply(parse_money)

# Distribution checks (EDA)
user_df['Budget'].describe()
property_df['Price'].describe()

# ===============================
# Feature engineering: price per square foot
# ===============================
property_df['price_per_sqft'] = (
    property_df['Price'] / property_df['Living Area (sq ft)']
)

# ===============================
# Log transforms (handle skewed distributions)
# ===============================
user_df['log_budget'] = np.log(user_df['Budget'])
property_df['log_price'] = np.log(property_df['Price'])
property_df['log_price_per_sqft'] = np.log(property_df['price_per_sqft'])
property_df['log_living_area'] = np.log(property_df['Living Area (sq ft)'])

# ===============================
# Text embedding setup (SBERT)
# ===============================
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Fix encoding artifacts like â€™
import ftfy
property_df['Qualitative Description'] = (
    property_df['Qualitative Description']
    .astype(str)
    .apply(ftfy.fix_text)
)

# Load pretrained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare text corpora
user_texts = user_df['Qualitative Description'].fillna('').tolist()
property_texts = property_df['Qualitative Description'].fillna('').tolist()

# Compute embeddings once (expensive step)
user_embeddings = model.encode(user_texts, show_progress_bar=True)
property_embeddings = model.encode(property_texts, show_progress_bar=True)

# Cosine similarity helper
def text_similarity(user_idx, property_idx):
    user_emb = user_embeddings[user_idx].reshape(1, -1)
    property_emb = property_embeddings[property_idx].reshape(1, -1)
    return cosine_similarity(user_emb, property_emb).flatten()[0]

# ===============================
# EDA support: keyword vocabulary
# ===============================
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', min_df=2)
cv.fit(user_df['Qualitative Description'])
cv.get_feature_names_out()[:50]

# ===============================
# Amenity extraction (rule-based)
# ===============================
AMENITY_KEYWORDS = {
    'outdoor_space': ['backyard', 'garden', 'deck', 'dock'],
    'premium_kitchen': ['kitchen', 'chef', 'appliances', 'island'],
    'entertaining_space': ['living', 'dining', 'entertaining', 'hosting', 'gatherings'],
    'high_ceilings': ['high ceilings', 'ceilings', 'beams', 'exposed'],
    'modern_design': ['modern', 'contemporary', 'design'],
    'cozy_feel': ['cozy', 'charming', 'ambiance'],
    'premium_segment': ['luxurious', 'estate', 'expansive']
}

import re

# Extract amenity set from free text
def extract_amenities(text):
    if pd.isna(text):
        return set()
    
    text = text.lower()
    found = set()
    
    for amenity, keywords in AMENITY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                found.add(amenity)
                break
    
    return found

user_df['amenity_set'] = user_df['Qualitative Description'].apply(extract_amenities)
property_df['amenity_set'] = property_df['Qualitative Description'].apply(extract_amenities)

# ===============================
# Amenity similarity (Jaccard)
# ===============================
def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

# ===============================
# Core match score computation
# ===============================
def compute_match_score(user, prop, user_idx, prop_idx):

    # Hard feasibility constraints
    if prop['Price'] > user['Budget']:
        return None
    if prop['Bedrooms'] < user['Bedrooms']:
        return None

    # Price compatibility (log-scaled)
    price_diff = abs(user['log_budget'] - prop['log_price'])
    price_sim = np.exp(-price_diff)

    # Area similarity (contextual, not dominant)
    area_diff = abs(
        np.log(prop['Living Area (sq ft)']) -
        np.log(user.get('Preferred Area', prop['Living Area (sq ft)']))
    )
    area_sim = np.exp(-area_diff)

    # Semantic intent similarity
    text_sim = text_similarity(user_idx, prop_idx)

    # Amenity overlap
    amenity_sim = jaccard_similarity(
        user['amenity_set'],
        prop['amenity_set']
    )

    # Clamp scores to [0,1]
    text_sim = max(0, text_sim)
    amenity_sim = min(1, amenity_sim)
    price_sim = min(1, max(0, price_sim))
    area_sim = min(1, max(0, area_sim))

    # Weighted final score (business-tunable)
    final_score = (
        0.35 * text_sim +
        0.25 * amenity_sim +
        0.25 * price_sim +
        0.15 * area_sim
    )

    return final_score

# ===============================
# Generate all user–property matches
# ===============================
results = []

for u_idx, user in user_df.iterrows():
    for p_idx, prop in property_df.iterrows():
        score = compute_match_score(user, prop, u_idx, p_idx)
        if score is not None:
            results.append({
                'user_id': u_idx,
                'property_id': p_idx,
                'match_score': score
            })

results_df = pd.DataFrame(results)

# Rank within each user
results_df_sorted = results_df.sort_values(
    by=['user_id', 'match_score'],
    ascending=[True, False]
)

TOP_N = 3

top_matches = (
    results_df_sorted
    .groupby('user_id')
    .head(TOP_N)
    .reset_index(drop=True)
)

# Attach property & user metadata
top_matches = top_matches.merge(
    property_df,
    left_on='property_id',
    right_index=True,
    how='left'
)

top_matches = top_matches.merge(
    user_df,
    left_on='user_id',
    right_index=True,
    how='left',
    suffixes=('_property', '_user')
)

# ===============================
# Display results (tabular)
# ===============================
for user_id, group in top_matches.groupby('user_id'):
    print(f"\nUser {user_id} – Top Matches")
    print("-" * 60)

    display_cols = group[
        [
            'property_id',
            'match_score',
            'Price',
            'Bedrooms_property',
            'Bathrooms_property',
            'Living Area (sq ft)'
        ]
    ].rename(columns={
        'Bedrooms_property': 'Bedrooms',
        'Bathrooms_property': 'Bathrooms'
    })

    print(display_cols.to_string(index=False))

# ===============================
# Visualization: bar chart for one user
# ===============================
USER_ID = top_matches['user_id'].unique()[0]
user_results = top_matches[top_matches['user_id'] == USER_ID]

plt.figure()
plt.bar(
    user_results['property_id'].astype(str),
    user_results['match_score']
)
plt.xlabel('Property ID')
plt.ylabel('Match Score')
plt.title(f'Top Property Matches for user_id {USER_ID}')
plt.show()

# ===============================
# Explainability: feature contribution heatmap
# ===============================
user = user_df.loc[USER_ID]
rows = []

for _, row in top_matches[top_matches['user_id'] == USER_ID].iterrows():
    prop = property_df.loc[row['property_id']]
    
    text_sim = text_similarity(USER_ID, row['property_id'])
    amenity_sim = jaccard_similarity(user['amenity_set'], prop['amenity_set'])
    
    price_diff = abs(user['log_budget'] - prop['log_price'])
    price_sim = np.exp(-price_diff)
    
    area_diff = abs(
        np.log(prop['Living Area (sq ft)']) -
        np.log(user.get('Preferred Area', prop['Living Area (sq ft)']))
    )
    area_sim = np.exp(-area_diff)
    
    rows.append([text_sim, amenity_sim, price_sim, area_sim])

feature_matrix = np.array(rows)

plt.figure()
plt.imshow(feature_matrix)
plt.colorbar()
plt.xticks(range(4), ['Text', 'Amenities', 'Price', 'Area'])
plt.yticks(
    range(len(feature_matrix)),
    top_matches[top_matches['user_id'] == USER_ID]['property_id'].astype(str)
)
plt.title(f'Feature Contributions for user_id {USER_ID}')
plt.show()