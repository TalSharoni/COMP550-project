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

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

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
    Extracts the text data points from the input dataset.
    Returns the text and their sentiments.
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
    A heuristic to classify the sentences.
    """
    data = data.dropna(subset=["text"]).reset_index(drop=True)
    data["text"] = data["text"].astype(str)

    num_pos = 0
    num_neg = 0

    y = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        if not data.at[i, "text"].strip():
            continue

        pos_count = 0
        neg_count = 0
        for word in data.at[i, "text"].split():
            if word in pos_seed_words:
                pos_count += 1
            if word in neg_seed_words:
                neg_count += 1

        if pos_count - neg_count > 1:
            y[i] = 1
            num_pos += 1
        elif neg_count - pos_count > diff_threshold:
            y[i] = -1
            num_neg += 1

    print("Total positive reviews:", num_pos)
    print("Total negative reviews:", num_neg)

    data["sentiment"] = y
    return data


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

    high_conf_indices = np.where(np.max(conf_scores, axis=1) > conf_thresh)[0]
    high_conf_samples = unlabelled_data.iloc[high_conf_indices].copy()
    high_conf_samples["sentiment"] = y_pred[high_conf_indices]

    if len(high_conf_samples) > 0:
        print(f"Adding {len(high_conf_samples)} high-confidence predictions to seed data.")
        data = pd.concat([data, high_conf_samples], ignore_index=True)
        unlabelled_data = unlabelled_data.drop(high_conf_samples.index)

        positive_reviews = data[data["sentiment"] == 1]["text"]
        negative_reviews = data[data["sentiment"] == -1]["text"]

        global pos_seed_words, neg_seed_words
        pos_seed_words = update_seed(positive_reviews, pos_seed_words)
        neg_seed_words = update_seed(negative_reviews, neg_seed_words)

        print(f"Updated positive seed words: {len(pos_seed_words)}")
        print(f"Updated negative seed words: {len(neg_seed_words)}")

    else:
        print("No high-confidence predictions this round. Reducing confidence threshold.")
        conf_thresh -= 0.05

    return data, unlabelled_data, conf_thresh


def bootstrap(data, initial_seed_size=1000, chunksize=2000, conf_thresh=0.9):
    accuracies = []
    seed_data = data.iloc[:initial_seed_size].copy()
    unlabelled_data = data.iloc[initial_seed_size:].copy()
    seed_data = classify(seed_data)
    labeled_counts = [len(seed_data)]

    train(seed_data)

    iteration = 0
    while len(unlabelled_data) > 0 and conf_thresh >= 0.65:
        iteration += 1
        print(f"\nBootstrapping Iteration {iteration}...")
        print(f"Current Confidence Threshold: {conf_thresh:.2f}")

        chunk_data = unlabelled_data.iloc[:chunksize]
        remaining_data = unlabelled_data.iloc[chunksize:]

        chunk_data = classify(chunk_data)
        unlabelled_chunk_data = chunk_data[chunk_data["sentiment"] == 0]
        chunk_data = chunk_data[chunk_data["sentiment"] != 0]

        seed_data, unlabelled_chunk_data, conf_thresh = boot_train(
            seed_data, unlabelled_chunk_data, conf_thresh=conf_thresh
        )
        labeled_counts.append(len(seed_data))

        unlabelled_data = pd.concat([remaining_data, unlabelled_chunk_data], ignore_index=True)
        unlabelled_data = unlabelled_data.sample(frac=1).reset_index(drop=True)

        accuracy_seed = calculate_accuracy(seed_data)
        accuracy_chunk = calculate_accuracy(chunk_data)

        accuracies.append((accuracy_seed, accuracy_chunk))
        print(f"Accuracy on seed data after iteration {iteration}: {accuracy_seed * 100:.2f}%")
        print(f"Accuracy on unlabelled data after iteration {iteration}: {accuracy_chunk * 100:.2f}%")

    return seed_data, accuracies, labeled_counts


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
    plt.plot(range(1, len(unlabelled_accuracies) + 1), unlabelled_accuracies, label="New Data Accuracy", marker='o')

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Evolution Over Bootstrapping Iterations")
    plt.legend()
    plt.show()


def data_percent_plot(labeled_counts, total_count):
    percentages = [(count / total_count) * 100 for count in labeled_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(percentages) + 1), percentages, marker='o', color='green', label='Labeled Data (%)')
    plt.xlabel("Iterations")
    plt.ylabel("Percentage of Data Labeled")
    plt.title("Progress of Data Labeling Over Iterations")
    plt.grid()
    plt.legend()
    plt.show()


def plot_confusion(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=[1, -1])
    labels = ["Positive", "Negative"]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center", color="black")

    plt.xticks([0, 1], labels, fontsize=10)
    plt.yticks([0, 1], labels, fontsize=10)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def main():
    """
    runs throught the 25k reviews and updates the seed
    outputs the positive and negative word clouds
    outputs some example classifications
    outputs overall accruracy of model
    """
    # data = get_processed_data("aclImdb/train")
    # data = get_processed_data("us-airline-tweets")
    data = get_processed_data("ecommerce-clothing-reviews")
    data["sentiment"] = 0

    data, acc, labeled_counts = bootstrap(data)
    print(f"Final positive seed words: {len(pos_seed_words)}")
    print(f"Final negative seed words: {len(neg_seed_words)}")

    accuracy_plot(acc)

    total_reviews = len(data)
    data_percent_plot(labeled_counts, total_reviews)

    plot_confusion(data["true_label"], data["sentiment"])

    positive_reviews = data[data["sentiment"] == 1]["text"]
    negative_reviews = data[data["sentiment"] == -1]["text"]
    word_cloud(positive_reviews, pos_seed_words, 'Blues')
    word_cloud(negative_reviews, neg_seed_words, 'Reds')


if __name__ == '__main__':
    main()