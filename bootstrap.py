# load in data
import pandas as pd
import numpy as np
import os

# remove stop words and punctuation and html tags and lowercase
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

pos_seed_words = [
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "fantastic",
    "terrific",
    "wonderful",
    "superb",
    "brilliant",
]
neg_seed_words = [
    "bad",
    "terrible",
    "crap",
    "useless",
    "hate",
    "horrible",
    "awful",
    "worst",
    "boring",
    "disgusting",
]


def get_processed_data(data_path):
    """
    extracts the movie reviews from the input dataset
    returns the reviews and their sentiments
    (sentiments are only used for the first 1000, rest get discarded and only used for accuracy calculation)

    There are 50k unsupervised reviews for training and then 25k supervised reviews for testing that we can use
    """
    # load in the data
    data = pd.DataFrame(columns=["text"])

    # navigate to folder
    for file in os.listdir(data_path + "/pos"):
        if file.endswith(".txt"):
            with open(data_path + "/pos/" + file, "r") as f:
                data.loc[len(data)] = preprocess(f.read())

    for file in os.listdir(data_path + "/neg"):
        if file.endswith(".txt"):
            with open(data_path + "/neg/" + file, "r") as f:
                data.loc[len(data)] = preprocess(f.read())

    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def preprocess(text):
    """
    tokenize, stopword removal, pos tag (maybe other things)
    return the preprocessed list of sentences and their pos tags

    So far only removing stopwords, punctuation, and html tags and putting everything in lemmatized lowercase- not sure if we need to do more
    """
    # remove html tags
    text = re.sub("<[^<]+?>", " ", text)
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # remove stop words and lemmatize
    text = " ".join(
        [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(text)
            if word.lower() not in stop_words
        ]
    )

    return text.lower()


def classify(data, diff_threshold=4):
    """
    probablistic model to classify the sentences
    uses the seed with cosine similarity, tfidf and more things
    returns the classified sentences
    maybe we can use multiple classifiers and compare?
    """

    num_pos = 0
    num_neg = 0

    # create a target variable
    y = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        pos_count = 0
        neg_count = 0
        for word in data.loc[i, "text"].split():
            if word in pos_seed_words:
                pos_count += 1
            if word in neg_seed_words:
                neg_count += 1
        if pos_count - neg_count > diff_threshold:
            y[i] = 1
            num_pos += 1
        elif neg_count - pos_count > diff_threshold:
            y[i] = -1
            num_neg += 1

    print("total positive reviews:", num_pos)
    print("total negative reviews:", num_neg)

    # add the target variable to the data
    data["sentiment"] = y


def train(data, max_iterations=10, conf_thresh=0.9):
    """
    runs throught the sentences and updates the seed
    returns the updated seed
    """
    vectorizer = CountVectorizer()

    # create a decision tree classifier
    clf = DecisionTreeClassifier(max_depth=5)

    # separate seed set from unlabelled set
    seed_set = data[data["sentiment"] != 0]
    unlabelled_set = data[data["sentiment"] == 0]

    X = seed_set["text"]
    y = seed_set["sentiment"]

    for i in range(max_iterations):

        # vectorize the text
        vector_X = vectorizer.fit_transform(X)
        unlabelled_X = vectorizer.transform(unlabelled_set["text"])

        # train classifier
        clf.fit(vector_X, y)

        # predict labels for unlabelled set
        y_pred = clf.predict(unlabelled_X)

        # get confidence scores
        conf_scores = clf.predict_proba(unlabelled_X)
        print(
            "scores", conf_scores
        )  # this is the confidence score for each class and if it is 0, 1 it is determining the class immediately

        # get indices of high confidence predictions
        high_conf_indices = np.where(np.max(conf_scores, axis=1) > conf_thresh)[0]
        print("Number of high confidence predictions:", len(high_conf_indices))

        if len(high_conf_indices) == 0:
            print("No high confidence predictions left", i)
            break

        # add high confidence predictions to seed set
        X = np.concatenate((X, unlabelled_set.iloc[high_conf_indices]["text"]))
        y = np.concatenate((y, y_pred[high_conf_indices]))
        # remove high confidence predictions from unlabelled set
        unlabelled_set = unlabelled_set.drop(unlabelled_set.index[high_conf_indices])

        if unlabelled_set.shape[0] == 0:
            print("No more unlabelled data left")
            break


def main():
    """
    runs throught the 50k reviews and updates the seed
    outputs the positive and negative word clouds
    outputs some example classifications
    outputs overall accruracy of model
    """
    data = get_processed_data("train")
    classify(data)
    train(data)

    print("done")

    # seed = {}
    # seed is of format (word, pos tag) : (positive weight, negative weight)


main()
