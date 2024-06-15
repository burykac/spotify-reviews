import os
import string
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from check_kaggle_API import check_kaggle_api
from verify_installation import verify
from download_data import download
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# weryfikacja i pobranie danych
verify()
check_kaggle_api()
download()

# przygotowanie datasetu do pracy
csv_file_path = os.path.join('data', 'spotify_reviews.csv')

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print(df.head())
else:
    print(f"Plik {csv_file_path} nie istnieje.")

# klasa przechowująca dataframe
class DataLoader:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_data(self):
        return self.df

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def normalize_text(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def remove_special(self, text):
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)

    def tokenize_text(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, text_data):
        return self.vectorizer.fit_transform(text_data)

class Scaler:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform_scaler(self, numeric_data):
        return self.scaler.fit_transform(numeric_data)

# Czyszczenie danych
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("Dane po czyszczeniu")
print(df.head())

# Normalizacja
text_processor = TextProcessor()

def process_text(text):
    normalized_text = text_processor.normalize_text(text)
    removed_punctuation = text_processor.remove_punctuation(normalized_text)
    removed_special = text_processor.remove_special(removed_punctuation)
    tokens = text_processor.tokenize_text(removed_special)
    tokens_removed_stopwords = text_processor.remove_stopwords(tokens)
    stemmed_tokens = text_processor.stem_tokens(tokens_removed_stopwords)
    lemmatized_tokens = text_processor.lemmatize_tokens(stemmed_tokens)
    return ' '.join(lemmatized_tokens)

df['normalized_content'] = df['content'].apply(process_text)

print("Dane po normalizacji ocen")
print(df[['content', 'normalized_content']].head())

