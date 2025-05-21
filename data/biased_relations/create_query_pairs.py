import json

# Load the original JSON file
with open('BRXX(XXX)/BRX_extracted_relations.json', 'r') as f:
    data = json.load(f)

# Create a list of (first, third) pairs from the 'output' list
pairs = []
for item in data:
    output = item.get('output', [])
    if len(output) >= 3:
        pairs.append([output[0], output[2]])

# Write the pairs to a new JSON file
with open('query_pairs.json', 'w') as f:
    json.dump(pairs, f, indent=2)

print("New JSON file with output pairs created.")