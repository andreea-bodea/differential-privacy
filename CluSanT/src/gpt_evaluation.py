import os
import re
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
organization_id = os.getenv("OPENAI_ORG_ID")

# Load the dataset
df = pd.read_csv("../anonymized_texts.csv")
filtered_df = df.copy()

# Initialize new columns in the DataFrame
for col in ["Grammar", "Common Sense", "Coherence", "Cohesiveness"]:
    filtered_df[col] = None

# Create a new column called "Custom_ID" and populate it with sequential values
filtered_df["Custom_ID"] = [
    "request-{}".format(i) for i in range(1, len(filtered_df) + 1)
]

# Initialize the OpenAI client
client = OpenAI(organization=organization_id)

# System message for OpenAI API
system_message = (
    "Could you please evaluate the following passage for its grammar, common sense, "
    "coherence, and cohesiveness? Score it on a scale from 1 to 5, where 1 is the lowest "
    "(poor quality) and 5 is the highest (excellent quality). "
    "You should score based on these criteria:\n"
    "grammar: Are the sentences structured correctly?\n"
    "common sense: Does the content make logical sense in the real world?\n"
    "coherence: Do the ideas flow logically from one sentence to another?\n"
    "cohesiveness: Do all parts of the text come together in a unified whole?\n"
    "Please ONLY respond in JSON format with the only four keys 'grammar', 'common sense', "
    "'coherence', and 'cohesiveness', each with a score attached to them."
)


# Function to clean and process text
def clean_text(text):
    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


# Write requests to a JSONL file
with open("requests_custext.jsonl", "w") as file:
    for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        custom_id = row["Custom_ID"]
        comment = clean_text(row["anonymized_text"])
        user_message = f"passage: {comment}"

        json_object = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 500,
            },
        }

        json.dump(json_object, file)
        file.write("\n")

# Create a batch input file for OpenAI API
batch_input_file = client.files.create(
    file=open("requests_custext.jsonl", "rb"), purpose="batch"
)

# Create a batch job
batch_response = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "nightly CluSanT evaluating job"},
)

# Print the status of the batch job
# print(client.batches.retrieve(batch_response.id).status)

# Retrieve the content of the batch response
content = client.files.content(
    client.batches.retrieve(batch_response.id).output_file_id
)

# Append the response content to a JSONL file
with open("requests_output_custext.jsonl", "ab") as file:
    file.write(content.content)


# Function to parse JSON content from the API response
def parse_content(content):
    try:
        clean_content = content.replace("```json\n", "").replace("\n```", "").strip()
        parsed_content = json.loads(clean_content)
        if isinstance(parsed_content, list):
            parsed_content = {str(i): item for i, item in enumerate(parsed_content)}
        return {k.lower(): v for k, v in parsed_content.items()}
    except json.JSONDecodeError:
        return {}


# Load and update the DataFrame with the API response
data = []
with open("requests_output_custext.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

for item in data:
    custom_id = item["custom_id"]
    content = parse_content(
        item["response"]["body"]["choices"][0]["message"]["content"]
    )

    if custom_id in filtered_df["Custom_ID"].values:
        filtered_df.loc[filtered_df["Custom_ID"] == custom_id, "Grammar"] = content.get(
            "grammar"
        )
        filtered_df.loc[filtered_df["Custom_ID"] == custom_id, "Common Sense"] = (
            content.get("common sense")
        )
        filtered_df.loc[filtered_df["Custom_ID"] == custom_id, "Coherence"] = (
            content.get("coherence")
        )
        filtered_df.loc[filtered_df["Custom_ID"] == custom_id, "Cohesiveness"] = (
            content.get("cohesiveness")
        )
    else:
        print(f"Custom_ID {custom_id} not found in DataFrame.")

# Save the updated DataFrame to a CSV file
filtered_df.to_csv("../evaluated_texts.csv", index=False)

# Replace the None values with NaN for proper handling
filtered_df[["Grammar", "Common Sense", "Coherence", "Cohesiveness"]] = filtered_df[
    ["Grammar", "Common Sense", "Coherence", "Cohesiveness"]
].apply(pd.to_numeric, errors="coerce")

# Group by the specified columns and calculate the average for the specified columns
grouped_averages = (
    filtered_df.groupby(["epsilon", "num_cluster"])
    .agg(
        {
            "Grammar": "mean",
            "Common Sense": "mean",
            "Coherence": "mean",
            "Cohesiveness": "mean",
        }
    )
    .reset_index()
)

# Iterate over the grouped dataframe and print each combination and the average values
for index, row in grouped_averages.iterrows():
    combination = (row["epsilon"], row["num_cluster"])
    grammar_avg = row["Grammar"]
    common_sense_avg = row["Common Sense"]
    coherence_avg = row["Coherence"]
    cohesiveness_avg = row["Cohesiveness"]

# Save the results to a CSV file
grouped_averages.to_csv("evaluation_results.csv", index=False)
