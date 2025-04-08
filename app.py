import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- Example transactions ---
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Cola']
]

# --- Preprocess ---
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# --- Frequent Itemsets & Rules ---
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# --- Streamlit UI ---
st.title("Market Basket Analysis Dashboard")

st.header("Transactions")
for i, t in enumerate(transactions, start=1):
    st.markdown(f"**Transaction {i}:** {', '.join(t)}")

st.header("Support Threshold")
min_support = st.slider("Min Support", 0.1, 1.0, 0.4, 0.05)
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
st.dataframe(frequent_itemsets)

st.header("Association Rules")
min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Convert frozen sets to readable strings
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(x))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(x))

st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
