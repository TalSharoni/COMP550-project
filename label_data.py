import os
import pandas as pd

'''
This function takes in the tweet dataset csv file and separates it into pos and neg 
folders according to the sentiment the tweet is labeled as.
'''
def classifiy_tweets(input_file, output_folder):

    pos_folder = os.path.join(output_folder, "pos")
    neg_folder = os.path.join(output_folder, "neg")

    # create pos and neg folders
    os.makedirs(pos_folder, exist_ok=True)
    os.makedirs(neg_folder, exist_ok=True)

    df = pd.read_csv(input_file) # Load the dataset

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


'''
This function takes in the clothing reviews dataset csv file and separates it into pos and neg 
folders according to the rating that the review is labeled as. We consider reviews rated 1-2 as negative 
and reviews rated 4-5 as positive. 
'''
def classifiy_reviews(input_file, output_folder):

    pos_folder = os.path.join(output_folder, "pos")
    neg_folder = os.path.join(output_folder, "neg")

    # create pos and neg folders
    os.makedirs(pos_folder, exist_ok=True)
    os.makedirs(neg_folder, exist_ok=True)

    df = pd.read_csv(input_file)  # Load the dataset

    for index, row in df.iterrows():
        review = str(row["Review Text"])  # Review content
        rating = row["Rating"]  # Rating

        # separate pos and neg lableled tweets into separate folders
        if rating > 3: # rating of 4 or 5
            folder = pos_folder
        elif rating < 3: # rating of 1 or 2
            folder = neg_folder
        else:
            continue  # skip if neutral (rating of 3)

        filename = f"review_{index}.txt"
        file_path = os.path.join(folder, filename)
        
        # Write the review to a text file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(review)

    print(f"Processing complete. Files saved in '{output_folder}' folder.")


def main():
    
    clothing_reviews_csv = "ecommerce-clothing-reviews/clothing_reviews.csv"
    clothing_reviews_output_folder = "ecommerce-clothing-reviews"
    classifiy_reviews(clothing_reviews_csv, clothing_reviews_output_folder)
    
    tweets_csv = "us-airline-tweets/Tweets.csv"
    tweets_output_folder = "us-airline-tweets"
    # classifiy_tweets(tweets_csv, tweets_output_folder)

if __name__ == '__main__':
    main()

    