import json

# Load the original JSON file
with open('BR01(Age)/BR01_extracted_relations.json', 'r') as f:
    data = json.load(f)

different_relations = []
for item in data:
    output = item["output"]
    if output[1] not in different_relations:
        different_relations.append(output[1])
    print(f"<\"{output[0]}\", \"{output[1]}\", \"{output[2]}\">")
    
print(different_relations)