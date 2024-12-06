import json

# Load JSON data from a file
with open('./src/prep-data/sampled_qa_pairs.json', 'r') as file:
    data = json.load(file)

# Extract the questions
questions = [entry["question"] for entry in data]

# Print the questions
for question in questions:
    print(question)
