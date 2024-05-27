# Importing TensorFlow-related packages
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Importing standard utility packages
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

# Importing local modules for model and data preprocessing
from Model import KeywordExtractionModel
from data import preprocess_text

# Set up logging and environment settings to manage TensorFlow's verbosity and numpy print options
stop_words = set(stopwords.words('english'))
import pickle
import logging
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure pandas display settings for better data frame readability
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load data from a CSV file, parse it, and clean it
data = []
with open("wikidata.csv", "r") as file:
    for line in file:
        fields = line.split("\t")  # Assuming tab delimiter; adjust as necessary
        if len(fields) == 2:
            data.append(fields)
        else:
            pass  # Handling of irregular lines can be included here
df = pd.DataFrame(data, columns=["Sentence", "Keyword"])

# Convert columns to strings
df['Sentence'] = df['Sentence'].astype(str)
df['Keyword'] = df['Keyword'].astype(str)

# Define function to check for numbers in strings
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# Function to tokenize and filter keywords
def tag_keywords(all_keywords):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([all_keywords])
    return list(set(tokenizer.word_index.keys()))

# Data preprocessing using custom preprocessing function
df['Sentence'] = df['Sentence'].apply(lambda x: x.replace(" – TechCrunch", "") if x is not None else x)
df['Processed_Sentences'] = df['Sentence'].apply(preprocess_text)

# Apply keyword tagging to 'Keyword' column
df['Keyword'] = df['Keyword'].apply(tag_keywords)

# Tokenize sentences and prepare input for model
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(df['Processed_Sentences'].dropna())

# Create tokenized sequences for model training
sentence_column = []
keyword_column = []
for index, row in df.iterrows():
    tokens = tokenizer.texts_to_sequences([row['Processed_Sentences']])[0]
    token_words = [tokenizer.index_word[token] for token in tokens if token in tokenizer.index_word]
    keyword_flags = [1 if word in row['Keyword'] else 0 for word in token_words]
    sentence_column.append(row['Processed_Sentences'])
    keyword_column.append(keyword_flags)

# Prepare and pad sequences
X = pad_sequences(tokenizer.texts_to_sequences(sentence_column), padding="post", maxlen=30)
y = pad_sequences(keyword_column, padding="post", maxlen=30, value=0)

# Convert labels to categorical format
y = [to_categorical(seq, num_classes=2) for seq in y]

# Load and prepare GloVe embeddings
embeddings_index = {}
with open('glove.6B.200d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word, coefs = values[0], np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

# Prepare the embedding matrix
EMBEDDING_DIM = 200
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model initialization and training
AKE_model = KeywordExtractionModel(256, word_index, EMBEDDING_DIM, embedding_matrix, 0.001)
AKE_model.fit(X_train, np.array(y_train), batch_size=64, epochs=50, validation_split=0.1)
model_json = AKE_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
AKE_model.save_weights("model.h6")
pickle.dump(tokenizer, open("tokenizer.pickle", "wb"))

# Model evaluation
test_loss, test_accuracy = AKE_model.evaluate(X_test, np.array(y_test))
test_output = AKE_model.predict(X_test)
test_output = np.argmax(test_output, axis=-1)
flattened_actual = np.argmax(np.array(y_test), axis=-1).flatten()
flattened_output = test_output.flatten()
print("Classification Report", classification_report(flattened_actual, flattened_output))

# Function to extract keywords from input text
def extract_keywords(text):
    input_ = preprocess_text(text)
    input_seq = tokenizer.texts_to_sequences([input_])
    input_seq = pad_sequences(input_seq, padding="post", maxlen=30)
    output = AKE_model.predict(input_seq)
    predictions = np.argmax(output, axis=-1)[0]
    return [word for idx, word in enumerate(input_.split()) if predictions[idx] == 1 and word not in stop_words]

# Interactive loop to process input texts and extract keywords
try:
    while True:
        text = input("Paragraphs: ")
        if text.strip():
            print("Keywords of given text:", extract_keywords(text))
        else:
            print("No input provided, please try again.")
except KeyboardInterrupt:
    print("Process interrupted by user.")
