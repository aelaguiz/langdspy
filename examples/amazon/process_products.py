import json
import random
import sys
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read the JSONL file
with open(input_file, 'r') as file:
    data = [json.loads(line) for line in file]

# Deduplicate based on the title
unique_data = {item['title']: item for item in data}.values()

# Process each item
X = []
y = []
for item in unique_data:
    # Trim strings
    item['title'] = item['title'].strip() if item.get('title') else ''
    item['h1'] = item['h1'].strip() if item.get('h1') else ''
    item['product_copy'] = ' '.join([copy.strip() for copy in item.get('product_copy', [])])
    url = item['url']
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    print(f"URL: {url} Path: {path_parts}")
    try:
        if "dp" == path_parts[2]:
            item['slug'] = path_parts[1]
            item['product_id'] = path_parts[3]
            X.append({
                'title': item['title'],
                'h1': item['h1'],
                'product_copy': item['product_copy']
            })
            y.append(item['slug'])
        elif "dp" == path_parts[1]:
            item['product_id'] = path_parts[2]
            item['slug'] = None
        else:
            print(f"Unknown URL format: {url}")
    except:
        print(f"Failed to parse URL: {url}")
        continue

# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store the datasets
datasets = {
    'train': {'X': X_train, 'y': y_train},
    'test': {'X': X_test, 'y': y_test}
}

# Save the datasets to a single JSON file
with open(output_file, 'w') as file:
    json.dump(datasets, file, indent=2)

print(f"Datasets saved to {output_file}")