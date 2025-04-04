import os
import re
import pandas as pd

def parse_directory_name(dir_name):
    """Extract lambda_pos, lambda_neg, and geoopt values from directory name."""
    match = re.search(r'lambda_pos_([\d\.]+)', dir_name)
    lambda_pos = float(match.group(1)) if match else None
    
    match = re.search(r'lambda_neg_([\d\.]+)', dir_name)
    lambda_neg = float(match.group(1)) if match else None
    
    match = re.search(r'geoopt_(True|False)', dir_name)
    geoopt = match.group(1) == "True" if match else None
    
    return lambda_pos, lambda_neg, geoopt


def parse_log_file(log_path):
    """Extract dataset, epoch, val_ndcg, val_hr, test_ndcg, and test_hr from log.txt."""
    with open(log_path, 'r') as f:
        content = f.readlines()
    
    data = []
    for line in content:
        match = re.match(r'(\d+) \(([-\d\.]+), ([-\d\.]+)\) \(([-\d\.]+), ([-\d\.]+)\)', line.strip())
        if match:
            epoch, val_ndcg, val_hr, test_ndcg, test_hr = map(float, match.groups())
            epoch = int(epoch)
            data.append((epoch, val_ndcg, val_hr, test_ndcg, test_hr))
    
    return data



def collect_data(base_path, dataset='ml-1m'):
    """Scan directories, extract data, and return a DataFrame."""
    data = []
    
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        
        if os.path.isdir(dir_path) and dataset in dir_name:
            lambda_pos, lambda_neg, geoopt = parse_directory_name(dir_name)
            log_path = os.path.join(dir_path, 'log.txt')
            
            if os.path.exists(log_path):
                log_data = parse_log_file(log_path)
                for epoch, val_ndcg, val_hr, test_ndcg, test_hr in log_data:
                    data.append({
                        'dataset': dataset,
                        'epoch': epoch,
                        'val_ndcg': val_ndcg,
                        'val_hr': val_hr,
                        'test_ndcg': test_ndcg,
                        'test_hr': test_hr,
                        'lambda_pos': lambda_pos,
                        'lambda_neg': lambda_neg,
                        'geoopt': geoopt
                    })
    
    return pd.DataFrame(data)

# Example usage
base_path = "C:\\Users\\user\\Documents\\GitHub\\SASRec.pytorch\\"
df = collect_data(base_path, dataset='Video')

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filter data for epoch = 2000
# df_filtered = df[df['epoch'] == 2000]

# # Define the specific conditions for each box
# conditions = [
#     (df_filtered['lambda_pos'] == 0) & (df_filtered['lambda_neg'] == 0) & (df_filtered['geoopt'] == False),
#     (df_filtered['lambda_pos'] == 0.1) & (df_filtered['lambda_neg'] == 0.5) & (df_filtered['geoopt'] == False),
#     (df_filtered['lambda_pos'] == 0) & (df_filtered['lambda_neg'] == 0) & (df_filtered['geoopt'] == True)
# ]

# labels = [
#     "λ+ = 0, λ- = 0, geoopt = False",
#     "λ+ = 0.1, λ- = 0.5, geoopt = False",
#     "λ+ = 0, λ- = 0, geoopt = True"
# ]

# # Extract test_ndcg values for each condition
# boxplot_data = [df_filtered[cond]['test_ndcg'] for cond in conditions]

# # Plot the boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=boxplot_data)
# plt.xticks(ticks=range(len(labels)), labels=labels)
# plt.ylabel("Test NDCG")
# plt.title("Test NDCG for Different Parameter Settings at Epoch 2000")
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Filter data for epoch = 2000
df_filtered = df[df['epoch'] == 20]

# Define the specific conditions and labels
conditions_labels = [
    ((df_filtered['lambda_pos'] == 0) & (df_filtered['lambda_neg'] == 0) & (df_filtered['geoopt'] == False),
     "λ+ = 0, λ- = 0, geoopt = False"),
    ((df_filtered['lambda_pos'] == 0.01) & (df_filtered['lambda_neg'] == 0.05) & (df_filtered['geoopt'] == False),
     "λ+ = 0.1, λ- = 0.5, geoopt = False"),
    ((df_filtered['lambda_pos'] == 0) & (df_filtered['lambda_neg'] == 0) & (df_filtered['geoopt'] == True),
     "λ+ = 0, λ- = 0, geoopt = True")
]

# Construct a new DataFrame for plotting
plot_data = []
for condition, label in conditions_labels:
    subset = df_filtered[condition].copy()
    subset["Condition"] = label
    plot_data.append(subset)

df_plot = pd.concat(plot_data)

# Plot the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Condition", y="test_ndcg", data=df_plot)
plt.xticks(rotation=15)
plt.ylabel("Test NDCG")
plt.xlabel("Parameter Settings")
plt.title("Test NDCG for Different Parameter Settings at Epoch 2000")
plt.show()
