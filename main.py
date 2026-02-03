import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


user_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="User Data")
property_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="Property Data")

user_df.head()
property_df.head()
user_df.info()
property_df.info()

def parse_money(x):
    x = x.replace('$','').strip()
    if 'k' in x.lower():
        return float(x.lower().replace('k','')) * 1000
    return float(x)

user_df['Budget'] = user_df['Budget'].apply(parse_money)
property_df['Price'] = property_df['Price'].apply(parse_money)


user_df['Budget'].describe()
property_df['Price'].describe()



property_df['price_per_sqft'] = property_df['Price'] / property_df['Living Area (sq ft)']



user_df['log_budget'] = np.log(user_df['Budget'])
property_df['log_price'] = np.log(property_df['Price'])
property_df['log_price_per_sqft'] = np.log(property_df['price_per_sqft'])
property_df['log_living_area'] = np.log(property_df['Living Area (sq ft)'])



from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


import ftfy

property_df['Qualitative Description'] = (
    property_df['Qualitative Description']
    .astype(str)
    .apply(ftfy.fix_text)
)

model= SentenceTransformer('all-MiniLM-L6-v2')
user_texts = user_df['Qualitative Description'].fillna('').tolist()
property_texts = property_df['Qualitative Description'].fillna('').tolist()

user_embeddings = model.encode(user_texts, show_progress_bar=True)
property_embeddings = model.encode(property_texts, show_progress_bar=True)

def text_similarity(user_idx, property_idx):
    user_emb = user_embeddings[user_idx].reshape(1, -1)
    property_emb = property_embeddings[property_idx].reshape(1, -1)
    return cosine_similarity(user_emb, property_emb).flatten()[0]

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', min_df=2)
cv.fit(user_df['Qualitative Description'])

cv.get_feature_names_out()[:50]

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
user_df['amenity_set'].head()
property_df['amenity_set'].head()

def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def compute_match_score(user, prop, user_idx, prop_idx):
    # hard filters
    if prop['Price'] > user['Budget']:
        return None
    if prop['Bedrooms'] < user['Bedrooms']:
        return None

    # numeric similarity
    price_diff = abs(user['log_budget'] - prop['log_price'])
    price_sim = np.exp(-price_diff)

    area_diff = abs(
    np.log(prop['Living Area (sq ft)']) -
    np.log(user.get('Preferred Area', prop['Living Area (sq ft)']))
    )
    area_sim = np.exp(-area_diff)


    # text similarity
    text_sim = text_similarity(user_idx, prop_idx)

    # amenity similarity
    amenity_sim = jaccard_similarity(
        user['amenity_set'],
        prop['amenity_set']
    )
    text_sim = max(0, text_sim)  # remove negative cosine
    amenity_sim = min(1, amenity_sim)
    price_sim = min(1, max(0, price_sim))
    area_sim = min(1, max(0, area_sim))


    final_score = (
        0.35 * text_sim +
        0.25 * amenity_sim +
        0.25 * price_sim +
        0.15 * area_sim
    )

    return final_score
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

print(sorted(top_matches.columns.tolist()))

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
top_matches[['user_id', 'property_id', 'match_score']].head(10)
import joblib

joblib.dump(model, "match_model.pkl")
