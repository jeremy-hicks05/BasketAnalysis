# from flask import Flask
from flask import jsonify, render_template
from flask_cors import CORS
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#app = Flask(__name__, template_folder="templates")
#CORS(app)

# Sample transactions dataset
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Cola']
]

def process_transactions():
    """Encodes transactions, applies Apriori, and extracts association rules."""
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # Convert sets to lists for JSON serialization
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))

    return transactions, frequent_itemsets, rules

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/unique-items')
def get_unique_items():
    """Return a list of unique items in the transactions."""
    unique_items = sorted(set(item for transaction in transactions for item in transaction))
    return jsonify({"items": unique_items})

@app.route('/transactions')
def get_transactions():
    """Return transactions in JSON for SVG rendering."""
    return jsonify({"transactions": transactions})

@app.route('/frequent-itemsets')
def get_frequent_itemsets():
    print("GET /frequent-itemsets called")  # Debugging log
    _, frequent_itemsets, _ = process_transactions()  # Correctly unpack all 3 returned values
    return jsonify([
        {"itemset": ', '.join(itemset), "support": support}
        for itemset, support in zip(frequent_itemsets['itemsets'], frequent_itemsets['support'])
    ])


# @app.route('/association-rules')
# def get_association_rules():
#     print("GET /association-rules called")  # Debugging log
#     _, _, rules = process_transactions()  # Correctly unpack all 3 returned values
#     nodes = set()
#     links = []

#     for _, row in rules.iterrows():
#         antecedents = ', '.join(row['antecedents'])
#         consequents = ', '.join(row['consequents'])
#         nodes.add(antecedents)
#         nodes.add(consequents)
#         links.append({
#             "source": antecedents,
#             "target": consequents,
#             "confidence": row["confidence"]
#         })

#     return jsonify({
#         "nodes": list(nodes),
#         "links": links
#     })

@app.route('/association-rules')
def get_association_rules():
    _, _, rules = process_transactions()
    nodes = set()
    links = []

    for _, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        nodes.add(antecedents)
        nodes.add(consequents)
        links.append({
            "source": antecedents,
            "target": consequents,
            "support": round(row["support"], 2),
            "confidence": round(row["confidence"], 2),
            "lift": round(row["lift"], 2)
        })

    return jsonify({
        "nodes": list(nodes),
        "links": links
    })



# @app.route('/support-data')
# def get_support_data():
#     """Return support values for line chart & histogram."""
#     frequent_itemsets, _ = process_transactions()
#     return jsonify([
#         {"index": i, "support": support}
#         for i, support in enumerate(frequent_itemsets['support'])
#     ])

@app.route('/support-data')
def get_support_data():
    print("GET /support-data called")  # Debugging log
    _, frequent_itemsets, _ = process_transactions()  # Correctly unpack all 3 returned values
    return jsonify([
        {"index": i, "support": support}
        for i, support in enumerate(frequent_itemsets['support'])
    ])


if __name__ == '__main__':
    app.run(debug=True)
