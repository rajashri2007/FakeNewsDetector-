import pandas as pd

# Read Excel file
df = pd.read_excel("dataset/HinFakeNews-V1.xlsx")

# Show actual columns
print("Columns found:", df.columns.tolist())

# Combine TITLE and CONTENT
df["text"] = (
    df["TITLE"].fillna("").astype(str)
    + " "
    + df["CONTENT"].fillna("").astype(str)
)

# BOOL:
# 0 = Fake
# 1 = Real
#
# Our model:
# 0 = Real
# 1 = Fake
df["label"] = df["BOOL"].map({0: 1, 1: 0})

# Keep required columns
output = df[["text", "label"]]

# Remove empty text
output = output[output["text"].str.strip() != ""]

# Remove invalid labels
output = output.dropna(subset=["label"])

# Save CSV
output.to_csv("dataset/fake_news.csv", index=False)

print("Dataset converted successfully!")
print("Total rows:", len(output))
print("\nLabel distribution:")
print(output["label"].value_counts())