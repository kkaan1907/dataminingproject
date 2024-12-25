import numpy as np
import pandas as pd
import collections
from itertools import permutations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

df = pd.read_csv("Groceries data.csv")

user_id = df['Member_number'].unique()
items = [list(df.loc[df['Member_number'] == id, 'itemDescription']) for id in user_id]
#print(items[0])


TE = TransactionEncoder()
TE.fit(items)
item_transformed = TE.transform(items)
item_matrix = pd.DataFrame(item_transformed, columns = TE.columns_)
#print(item_matrix.head())

freq_items = apriori(item_matrix, min_support=0.01, use_colnames=True, max_len=2)
freq_items.sort_values(by = "support", ascending = False)

rules = association_rules(freq_items, metric = "confidence", min_threshold = 0, num_itemsets=0)
#print(rules)

freq_items = apriori(item_matrix, min_support=0.01, use_colnames=True, max_len=5)
freq_items.sort_values(by = "support", ascending = False)

def zhangs_rule(rules2):
    rule_support = rules2['support'].copy()
    rule_ante = rules2['antecedent support'].copy()
    rule_conseq = rules2['consequent support'].copy()
    num = rule_support - (rule_ante * rule_conseq)
    denom = np.max((rule_support * (1 - rule_ante).values,
                          rule_ante * (rule_conseq - rule_support).values), axis = 0)
    return num / denom

rules_zhangs_list = zhangs_rule(rules)
rules = rules.assign(zhang = rules_zhangs_list)
#print(rules.head())

rules_eda= rules
rules_eda['antecedents'] = rules['antecedents'].apply(lambda a: ', '.join(list(a)))
rules_eda['consequents'] = rules['consequents'].apply(lambda a: ', '.join(list(a)))


rules1=rules_eda
rules1['scorelftlvg']=(rules1['lift']*0.5 +rules1['leverage']*0.5)*100
rules1['scoreconcnf']=(rules1['confidence']*0.5 +rules1['conviction']*0.5)*100
rules1=rules1.sort_values(['scorelftlvg', 'scoreconcnf'], ascending=False)
print(rules1)

def suggestitem(k):
    df = pd.DataFrame()
    rules_eda = rules1[(rules1['lift'] > 1) & (rules1['zhang'] > 0) & (rules1['leverage'] > 0)]
    if len(rules_eda[rules_eda['antecedents'] == k]) != 0:
        dataf = rules_eda[rules_eda['antecedents'] == k][
            ['antecedents', 'consequents', 'lift', 'leverage', 'scorelftlvg']]
        dat = dataf[['antecedents', 'consequents', 'lift', 'scorelftlvg']]
        dat = dat.sort_values(['lift'], ascending=False)
        daf = dataf[['antecedents', 'consequents', 'leverage', 'scorelftlvg']]
        daf = daf.sort_values(['leverage'], ascending=False)
        da = dataf[['antecedents', 'consequents', 'scorelftlvg']]
        da = da.sort_values(['scorelftlvg'], ascending=False)
        its = (da['consequents'].head(5)).to_list()
        itmlst = (daf['consequents'].head(3)).to_list() + (dat['consequents'].head(3)).to_list()
        itmlst = list(set(itmlst))
        score = []
        for itm in itmlst:
            f = daf[daf['consequents'] == itm]['scorelftlvg'].item()
            score.append(f)

        frames = [itmlst, score]
        df['nextitem'] = itmlst
        df['Score'] = score
        # itm=', '.join(itmlst)
        for i in range(len(df)):
            print(k, "is frequently bought with ", itmlst[i], " with score ", "{:.2f}".format(score[i]))


grp1 = ['coffee', 'frozen vegetables', 'chicken', 'white bread',
        'cream cheese ', 'chocolate', 'dessert', 'napkins', 'berries',
        'hamburger meat', 'UHT-milk', 'onions', 'salty snack', 'waffles',
        'long life bakery product', 'sugar', 'butter milk', 'ham', 'meat']
for i in range(len(grp1)):
    print(suggestitem(grp1[i]))



