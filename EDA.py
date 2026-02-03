import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
user_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="User Data")
property_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="Property Data")


def clean_price_numeric(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float)):
        return float(x)

    x = str(x).strip().lower()
    x = x.replace("$", "").replace(",", "")

    if x.endswith("k"):
        return float(x[:-1]) * 1000

    return float(x)

property_df['Price'] = property_df['Price'].apply(clean_price_numeric)
user_df['Budget'] = user_df['Budget'].apply(clean_price_numeric)


property_df['price_per_sqft'] = (
    property_df['Price'] / property_df['Living Area (sq ft)']
)


property_df = property_df.dropna(
    subset=['Price', 'Living Area (sq ft)', 'price_per_sqft']
)

user_df.head()
property_df.head()
user_df.info()
property_df.info()

sns.histplot(user_df['Budget'], bins=30, kde=True)
plt.title('Distribution of User Budgets')
plt.xlabel('Budget (in thousands)')
plt.ylabel('Frequency')
plt.show()

plt.plot(user_df['Bedrooms'], user_df['Budget'], 'o')
plt.title('User Budget vs Desired Bedrooms')
plt.xlabel('Budget (in thousands)')
plt.ylabel('Number of Bedrooms')
plt.show()

plt.plot(property_df['Bedrooms'], property_df['Price'], 'o', color='orange')
plt.title('Property Price vs Bedrooms')
plt.xlabel('Price (in thousands)')
plt.ylabel('Number of Bedrooms')
plt.show()

plt.plot(property_df['Price'], property_df['Living Area (sq ft)'], 'o', color='green')
plt.title('Property Price vs Living Area')
plt.xlabel('Price (in thousands)')
plt.ylabel('Living Area (sq ft)')
plt.show()

sns.histplot(property_df['Price'], bins=30, kde=True, color='purple')
plt.title('Distribution of Property Prices')
plt.xlabel('Price (in thousands)')
plt.ylabel('Frequency')
plt.show()
plt.plot(property_df['Living Area (sq ft)'], property_df['price_per_sqft'], 'o', color='red')
plt.title('Price per Sq Ft vs Living Area')
plt.xlabel('Living Area (sq ft)')
plt.ylabel('Price per Sq Ft (in thousands)')
plt.show()

property_df['budget_compatible'] = property_df['Price'] <= user_df['Budget'].median()
plt.hist(property_df[property_df['budget_compatible']]['Price'], bins=30, alpha=0.5, label='Price', color='blue')
plt.hist(user_df['Budget'], bins=30, alpha=0.5, label='User Budgets', color='orange')
plt.title('Property Prices vs User Budgets')
plt.xlabel('log(Price/Budget) (in thousands)')
plt.ylabel('Frequency')
plt.legend()
plt.show()