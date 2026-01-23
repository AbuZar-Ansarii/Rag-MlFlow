import os
import json

with open(r'E:\pandas\MLOPS\RAG Mlflow\prompts\prompts.json', 'r') as f:
    prompts = json.load(f)

for prompt in prompts['research_assistant_prompts']:
    print(prompt)


