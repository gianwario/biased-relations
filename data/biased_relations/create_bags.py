import json
import os

# Input and output file paths
input_path = 'BR01(Age)/BR01_prompts.json'  # your actual file
output_path = 'BR01_bags.json'

# Extract the filename without extension to use as key
file_key = os.path.splitext(os.path.basename(input_path))[0]

# Read the input JSON
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect only the outputs lists
outputs_collected = [item["outputs"] for item in data]

# Build the final output structure
final_output = {file_key: outputs_collected}

# Save the result
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(f"Done! Output written to {output_path}")
