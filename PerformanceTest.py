import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import numpy as np
import time
import psutil
from scipy.sparse import csr_matrix
# Read data
data = pd.read_csv('basket.csv', header=None)
data = data[0].apply(lambda x: x.split(','))

# Transaction Encoder
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Function to monitor memory usage
def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2  # Memory in MB

# Run Apriori algorithm
def run_apriori(df, min_support):
    start_time = time.time()
    initial_memory = memory_usage()
    apriori_frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    apriori_time = time.time() - start_time
    final_memory = memory_usage()
    memory_diff = final_memory - initial_memory
    return apriori_frequent_itemsets, apriori_time, memory_diff


def eclat(df, min_support=0.01, use_colnames=True):
    # Convert the DataFrame to a sparse matrix (csr_matrix)
    sparse_df = csr_matrix(df.values)

    # Calculate item support (frequency of each item across transactions)
    item_support = sparse_df.mean(axis=0).A1  # A1 converts from matrix to array
    frequent_items = item_support[item_support >= min_support]

    # Create initial itemsets with single items
    results = []
    for idx, item in enumerate(df.columns):
        if item_support[idx] >= min_support:
            results.append({
                'support': item_support[idx],
                'itemsets': frozenset([item])
            })

    k = 2
    while True:
        candidate_itemsets = []
        # Iterate over pairs of frequent itemsets from the previous step
        for i, itemset_i in enumerate(results):
            for j, itemset_j in enumerate(results[i + 1:], start=i + 1):
                new_itemset = itemset_i['itemsets'].union(itemset_j['itemsets'])

                if len(new_itemset) == k:
                    # Use bitwise AND to find the support of the new itemset
                    cols = [df.columns.get_loc(item) for item in new_itemset]
                    bitwise_support = np.bitwise_and.reduce(sparse_df[:, cols].toarray(), axis=1).mean()

                    if bitwise_support >= min_support:
                        candidate_itemsets.append({
                            'support': bitwise_support,
                            'itemsets': new_itemset
                        })

        if not candidate_itemsets:
            break

        results.extend(candidate_itemsets)
        k += 1

    result_df = pd.DataFrame(results)
    if use_colnames:
        result_df['itemsets'] = result_df['itemsets'].apply(list)

    return result_df

def run_eclat(df, min_support):
    start_time = time.time()
    initial_memory = memory_usage()
    eclat_frequent_itemsets = eclat(df, min_support=min_support)
    eclat_time = time.time() - start_time
    final_memory = memory_usage()
    memory_diff = final_memory - initial_memory
    return eclat_frequent_itemsets, eclat_time, memory_diff

# Compare Apriori and Eclat
def compare_algorithms(df, min_support):
    print("Performance Comparison Started...")

    # Run Apriori
    apriori_frequent_itemsets, apriori_time, apriori_memory = run_apriori(df, min_support)
    print(f"\nApriori Execution Time: {apriori_time:.4f} seconds")
    print(f"Apriori Memory Usage: {apriori_memory:.4f} MB")
    print(f"Apriori Found Frequent Itemsets: {len(apriori_frequent_itemsets)}")

    # Run ECLAT
    eclat_frequent_itemsets, eclat_time, eclat_memory = run_eclat(df, min_support)
    print(f"\nEclat Execution Time: {eclat_time:.4f} seconds")
    print(f"Eclat Memory Usage: {eclat_memory:.4f} MB")
    print(f"Eclat Found Frequent Itemsets: {len(eclat_frequent_itemsets)}")

    # Print Top 10 results
    print("\nApriori Top 10 Highest Support Itemsets:")
    print(apriori_frequent_itemsets.sort_values('support', ascending=False).head(10))

    print("\nEclat Top 10 Highest Support Itemsets:")
    print(eclat_frequent_itemsets.sort_values('support', ascending=False).head(10))

# Main function
def main():
    min_support = 0.01
    compare_algorithms(df, min_support)

if __name__ == '__main__':
    main()
