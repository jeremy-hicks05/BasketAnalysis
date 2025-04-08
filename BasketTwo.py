
print('test')

# Install mlxtend if not already available
#!pip install mlxtend

# Import necessary libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

print("Libraries imported successfully!")

# Example list of transactions (each transaction is a list of items bought together)
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Cola']
]

# Preview the example transactions
#for i, trans in enumerate(transactions, start=1):
    #print(f"Transaction {i}: {trans}")

# Initialize the TransactionEncoder
te = TransactionEncoder()

# Transform the list of transactions into an array of booleans
te_array = te.fit(transactions).transform(transactions)

# Convert to pandas DataFrame for easier handling
df = pd.DataFrame(te_array, columns=te.columns_)

# Show the one-hot encoded dataframe (first few rows)
print(df.head())

# Apply the Apriori algorithm to find frequent itemsets with min support of 0.4 (40%)
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Display the frequent itemsets found
print(frequent_itemsets)

from matplotlib import pyplot as plt
frequent_itemsets['support'].plot(kind='line', figsize=(8, 4), title='support')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

from matplotlib import pyplot as plt
frequent_itemsets['support'].plot(kind='hist', bins=20, title='support')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

# Generate association rules from the frequent itemsets with a minimum confidence of 0.5 (50%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the resulting rules
print(rules)

# Sort rules by highest lift to see the strongest associations at the top
print(rules.sort_values('lift', ascending=False).head(10))

# Sort rules by highest lift to see the strongest associations at the top
print(rules.sort_values('lift', ascending=False).head(10))

print(rules[rules['antecedents'].apply(lambda x: 'Bread' in x)])


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
