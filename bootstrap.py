# load in data
import pandas as pd
import numpy as np
import os

# remove stop words and punctuation and html tags and lowercase
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    data = pd.DataFrame(columns=["text", "true_label"])

    # pos reviews
    for file in os.listdir(data_path + "/pos"):
        if file.endswith(".txt"):
            with open(data_path + "/pos/" + file, "r") as f:
                data.loc[len(data)] = [preprocess(f.read()), 1]

    # neg reviews
    for file in os.listdir(data_path + "/neg"):
        if file.endswith(".txt"):
            with open(data_path + "/neg/" + file, "r") as f:
                data.loc[len(data)] = [preprocess(f.read()), -1]

    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def preprocess(text):
    """
    removing stopwords, punctuation, and html tags and putting everything in lemmatized lowercase
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


def update_seed(reviews, seed_words, overlaps=100):
    word_counts = {}

    for text in reviews:
        words = set(word_tokenize(text))
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    for word, count in word_counts.items():
        if count >= overlaps and word not in seed_words:
            seed_words.append(word)

    return list(set(seed_words))


def train(data, overlaps=100):
    global pos_seed_words, neg_seed_words

    data["sentiment"] = data["true_label"]
    positive_reviews = data[data["true_label"] == 1]["text"]
    negative_reviews = data[data["true_label"] == -1]["text"]

    pos_seed_words = update_seed(positive_reviews, pos_seed_words, overlaps)
    neg_seed_words = update_seed(negative_reviews, neg_seed_words, overlaps)

    print(f"Updated positive seed words: {len(pos_seed_words)}")
    print(f"Updated negative seed words: {len(neg_seed_words)}")

    return pos_seed_words, neg_seed_words


def classify(data, diff_threshold=4):
    """
    a heuristic to classify the sentences
    returns the classified sentences
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


def boot_train(data, unlabelled_data, conf_thresh=0.9, max_depth=5):
    """
    runs throught the sentences and updates the seed
    returns the updated seed
    """
    vectorizer = CountVectorizer()
    classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=5)

    X_seed = vectorizer.fit_transform(data["text"])
    y_seed = data["sentiment"]
    classifier.fit(X_seed, y_seed)

    X_unlabelled = vectorizer.transform(unlabelled_data["text"])
    y_pred = classifier.predict(X_unlabelled)
    conf_scores = classifier.predict_proba(X_unlabelled)

    unlabelled_data["sentiment"] = y_pred

    high_conf_indices = np.where(np.max(conf_scores, axis=1) > conf_thresh)[0]
    high_conf_samples = unlabelled_data.iloc[high_conf_indices]

    if len(high_conf_samples) > 0:
        print(f"Adding {len(high_conf_samples)} high-confidence predictions to seed data.")
        data = pd.concat([data, high_conf_samples])
        unlabelled_data = unlabelled_data.drop(high_conf_samples.index)
    else:
        print("No high-confidence predictions this round.")

    positive_reviews = data[data["sentiment"] == 1]["text"]
    negative_reviews = data[data["sentiment"] == -1]["text"]

    global pos_seed_words, neg_seed_words
    pos_seed_words = update_seed(positive_reviews, pos_seed_words)
    neg_seed_words = update_seed(negative_reviews, neg_seed_words)

    print(f"Updated positive seed words: {len(pos_seed_words)}")
    print(f"Updated negative seed words: {len(neg_seed_words)}")

    return data, unlabelled_data


def bootstrap(data, initial_seed_size=1000, chunksize=1000, conf_thresh=0.9):
    accuracies = []
    seed_data = data.iloc[:initial_seed_size]
    unlabelled_data = data.iloc[initial_seed_size:]

    train(seed_data)

    iteration = 0
    while len(unlabelled_data) > 0:
        iteration += 1
        print(f"Bootstrapping iteration {iteration}...")

        chunk_data = unlabelled_data.iloc[:chunksize]
        remaining_unlabelled_data = unlabelled_data.iloc[chunksize:]

        seed_data, unlabelled_data = boot_train(seed_data, chunk_data, conf_thresh=conf_thresh)
        data.update(seed_data)
        print(seed_data, data)
        accuracy_seed = calculate_accuracy(seed_data)
        accuracy_unlabelled = calculate_accuracy(unlabelled_data)

        accuracies.append((accuracy_seed, accuracy_unlabelled))
        print(f"Accuracy on seed data after iteration {iteration}: {accuracy_seed * 100:.2f}%")
        print(f"Accuracy on unlabelled data after iteration {iteration}: {accuracy_unlabelled * 100:.2f}%")

        unlabelled_data = remaining_unlabelled_data

    return accuracies


def word_cloud(text, seed, color):
    words = [word for word in word_tokenize(" ".join(text)) if word in seed]
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=color).generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def calculate_accuracy(data):
    correct_predictions = (data["true_label"] == data["sentiment"]).sum()
    accuracy = correct_predictions / len(data)
    return accuracy


def accuracy_plot(acc):
    seed_accuracies, unlabelled_accuracies = zip(*acc)
    plt.plot(range(1, len(seed_accuracies) + 1), seed_accuracies, label="Seed Data Accuracy", color='blue', marker='o')
    plt.plot(range(1, len(unlabelled_accuracies) + 1), unlabelled_accuracies, label="Unlabelled Data Accuracy", marker='o')

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Evolution Over Bootstrapping Iterations")
    plt.legend()
    plt.show()


def main():
    """
    runs throught the 25k reviews and updates the seed
    outputs the positive and negative word clouds
    outputs some example classifications
    outputs overall accruracy of model
    """
    data = get_processed_data("aclImdb/train")
    data["sentiment"] = 0

    acc = bootstrap(data)

    accuracy_plot(acc)

    positive_reviews = data[data["sentiment"] == 1]["text"]
    negative_reviews = data[data["sentiment"] == -1]["text"]
    word_cloud(positive_reviews, pos_seed_words, 'Blues')
    word_cloud(negative_reviews, neg_seed_words, 'Reds')


if __name__ == '__main__':
    main()