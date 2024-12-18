import os
import pandas as pd


input_file = "us-airline-tweets/Tweets.csv"  # Update with your dataset's path
output_folder = "us-airline-tweets"

pos_folder = os.path.join(output_folder, "pos")
neg_folder = os.path.join(output_folder, "neg")

os.makedirs(pos_folder, exist_ok=True)
os.makedirs(neg_folder, exist_ok=True)

# Load the us-airline tweet dataset
df = pd.read_csv(input_file)

for index, row in df.iterrows():
    tweet = row["text"]  # Tweet content
    sentiment = row["airline_sentiment"].lower()  # Sentiment label

    # separate pos and neg lableled tweets into separate folders
    if sentiment == "positive":
        folder = pos_folder
    elif sentiment == "negative":
        folder = neg_folder
    else:
        continue  # skip if neutral

    filename = f"tweet_{index}.txt"
    file_path = os.path.join(folder, filename)
    
    # Write the tweet to a text file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(tweet)

print(f"Processing complete. Files saved in '{output_folder}' folder.")
