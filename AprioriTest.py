import numpy as np
import pandas as pd

# For data analysis
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

groceries_df = pd.read_csv('Groceries data.csv')
basket_df = pd.read_csv('basket.csv')

basket_df.fillna('NA', inplace=True)
basket_df_list = basket_df.values.tolist()

# Removing 'NA' from each list
for i in range(len(basket_df_list)):
    basket_df_list[i] = [x for x in basket_df_list[i] if not x=='NA']

te = TransactionEncoder()
te_ary = te.fit(basket_df_list).transform(basket_df_list)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori Algorithm

frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1, num_itemsets=2)

frequent_itemsets.head(10)

print(rules.sort_values(by='confidence', ascending=False).head(10))

