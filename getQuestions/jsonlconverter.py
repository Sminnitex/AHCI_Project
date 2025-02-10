import json

# Read the JSONL file and convert it into a list of dictionaries
questions = []
with open("dev_rand_split.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        questions.append(json.loads(line))

# Save as a standard JSON file for easier use in React
with open("questions.json", "w", encoding="utf-8") as outfile:
    json.dump(questions, outfile, indent=4)