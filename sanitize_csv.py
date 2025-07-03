import pandas as pd
from SanText.sanitize_one_sentence import SanTextBatchProcessor

# Try to read the CSV, skipping bad lines
try:
    df = pd.read_csv("mental-health-blog-50(in).csv", on_bad_lines='skip')  # pandas >= 1.3
except TypeError:
    df = pd.read_csv("mental-health-blog-50(in).csv", error_bad_lines=False)  # pandas < 1.3

print("Columns in CSV:", list(df.columns))
print("First 5 rows:")
print(df.head())

# TODO: Update this to the correct column name after checking the printout
text_column = "post"

# Initialize the processor
processor = SanTextBatchProcessor()

# Sanitize each sentence
sanitized_sentences = []
for sentence in df[text_column]:
    sanitized = processor.sanitize(str(sentence), method="SanText")
    sanitized_sentences.append(sanitized)

# Add sanitized sentences to the DataFrame
df['sanitized'] = sanitized_sentences

# Save to a new CSV
df.to_csv("mental-health-blog-50_sanitized.csv", index=False)
print("Sanitization complete. Output saved to mental-health-blog-50_sanitized.csv") 