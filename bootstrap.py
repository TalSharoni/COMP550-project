import os
import random

def get_data(input):
  """
  extracts the movie reviews from the input dataset
  returns the reviews and their sentiments (1 is positive sentiment, 0 is negative sentiment)
  (sentiments are only used for the first 1000, rest get discarded and only used for accuracy calculation)
  """
  train_data_pos = os.path.join(input, 'train/pos')
  train_data_neg = os.path.join(input, 'train/neg')

  reviews = []
  sentiments = []

  # Get positive reviews
  for filename in os.listdir(train_data_pos):
    filepath = os.path.join(train_data_pos, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
      reviews.append(file.read())
      sentiments.append(1)
  
  # Get negative reviews
  for filename in os.listdir(train_data_neg):
    filepath = os.path.join(train_data_neg, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
      reviews.append(file.read())
      sentiments.append(0)  

  # Shuffle data
  data = list(zip(reviews, sentiments))
  random.seed(42)
  random.shuffle(data) 

  # Split into training and testing sets
  reviews, sentiments = zip(*data)
  train_reviews = list(reviews[:1000])
  train_sentiments = list(sentiments[:1000])
  test_reviews = list(reviews[1000:])
  test_sentiments = list(sentiments[1000:])

  return train_reviews, train_sentiments, test_reviews, test_sentiments

def preprocess(sentences):
  """
  tokenize, stopword removal, pos tag (maybe other things)
  return the preprocessed list of sentences and their pos tags
  """
  pass

def classify(sentences, seed):
  """
  probablistic model to classify the sentences
  uses the seed with cosine similarity, tfidf and more things
  returns the classified sentences
  maybe we can use multiple classifiers and compare?
  """
  pass

def train(sentences, seed, size=1000):
  """
  runs throught the sentences and updates the seed
  returns the updated seed
  """
  pass

def main():
  """
  runs throught the 50k reviews and updates the seed
  outputs the positive and negative word clouds
  outputs some example classifications
  outputs overall accruracy of model
  """
  # note: download the movie review dataset and put it in the same directory as this project
  train_reviews, train_sentiments, test_reviews, test_sentiments = get_data('aclImdb')
  seed = {}
  # seed is of format (word, pos tag) : (positive weight, negative weight)

if __name__ == '__main__':
  main()