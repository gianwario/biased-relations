import json
import re

# Load the original JSON structure
with open('BR09(Socioeconomic)/BR09_prompts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, item in enumerate(data):
    # Extract group term from input
    input_triple = item['input']
    match = re.match(r'<\s*(.*?),\s*.*?,\s*.*?>', input_triple)
    if not match:
        print(f"❌ Could not parse input triple: {input_triple}")
        continue
    group_term = match.group(1)

    for j, output in enumerate(item['outputs']):
        sentence, expected_group, _ = output

        if group_term != expected_group:
            print(f"⚠️  Mismatch in input/output group at item {i}, output {j}")
            print(f"   Input group: {group_term}")
            print(f"   Output group: {expected_group}")

        # Reconstruct by replacing [MASK] with group term
        if "[MASK]" not in sentence:
            print(f"❌ Missing [MASK] in output at item {i}, output {j}")
            print(f"   Sentence: {sentence}")
            continue

        reconstructed = sentence.replace("[MASK]", group_term)

        # Check if the group term appears once and only once
        count = reconstructed.count(group_term)
        if count != 1:
            print(f"❗ Suspicious [MASK] placement at item {i}, output {j}")
            print(f"   Sentence: {sentence}")
            print(f"   Reconstructed: {reconstructed}")
            print(f"   Occurrence count: {count}")