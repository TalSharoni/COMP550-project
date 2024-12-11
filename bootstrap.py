def get_data(input):
  """
  extracts the movie reviews from the input dataset
  returns the reviews and their sentiments 
  (sentiments are only used for the first 1000, rest get discarded and only used for accuracy calculation)
  """
  pass

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
  pass
  seed = {}
  # seed is of format (word, pos tag) : (positive weight, negative weight)

