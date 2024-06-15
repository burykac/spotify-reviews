import os
import string
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from check_kaggle_API import check_kaggle_api
from verify_installation import verify
from download_data import download
from scipy import sparse

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Weryfikacja i pobranie danych
verify()
check_kaggle_api()
download()

# Przygotowanie datasetu do pracy
csv_file_path = os.path.join('data', 'spotify_reviews.csv')

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print(df.head())
else:
    print(f"Plik {csv_file_path} nie istnieje.")

# Wstępna analiza danych
review_count = df['score'].value_counts()
order = [1, 2, 3, 4, 5]
review_count = review_count.reindex(order)
plt.bar(review_count.index, review_count.values)
plt.show()

avg_score = df['score'].sum() / len(df)
print(f"Średnia ocena aplikacji: {avg_score}")


# Klasa przechowująca DataFrame
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

    def process_text(self, text):
        normalized_text = self.normalize_text(text)
        removed_punctuation = self.remove_punctuation(normalized_text)
        removed_special = self.remove_special(removed_punctuation)
        tokens = self.tokenize_text(removed_special)
        tokens_removed_stopwords = self.remove_stopwords(tokens)
        if not tokens_removed_stopwords:
            return ""
        stemmed_tokens = self.stem_tokens(tokens_removed_stopwords)
        lemmatized_tokens = self.lemmatize_tokens(stemmed_tokens)
        return ' '.join(lemmatized_tokens)


# Czyszczenie danych
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("Dane po czyszczeniu")
print(df[['content', 'score']].head())

# Normalizacja
text_processor = TextProcessor()
df['normalized_content'] = df['content'].apply(text_processor.process_text)

print("Dane po normalizacji ocen")
df = df[df['normalized_content'].str.strip() != ""]

print(df[['content', 'score', 'normalized_content']].head(10))

if isinstance(df['normalized_content'].iloc[0], str):
    print("Dane są w odpowiednim formacie do wektoryzacji.")
else:
    raise ValueError("Kolumna 'normalized_content' nie zawiera str.")


# Klasa do wektoryzacji tekstu
class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, text_data):
        self.matrix = self.vectorizer.fit_transform(text_data)
        return self.matrix

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


# Wektoryzacja tekstu
vectorizer = Vectorizer()
df['lemmatized_content'] = df['normalized_content']

text_for_matrix = df['lemmatized_content'].tolist()

try:
    matrix = vectorizer.fit_transform(text_for_matrix)
    feature_names = vectorizer.get_feature_names_out()
    print("Wektoryzacja zakończona sukcesem.")
except ValueError as e:
    print(f"Błąd wektoryzacji: {e}")

# Sprawdzenie macierzy
if matrix.nnz == 0:
    print("Macierz zawiera tylko zera. Sprawdź przetwarzanie tekstu.")
else:
    print("Macierz zawiera wartości różne od zera.")

sparse.save_npz('data/tfidf_matrix.npz', matrix)
print("Macierz zapisana do pliku 'data/tfidf_matrix.npz'.")

sample_matrix = matrix[:20].toarray()
print(sample_matrix)

# Analiza sentymentu
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df['polarity'], df['subjectivity'] = zip(*df['content'].apply(analyze_sentiment))

print(df[['content', 'polarity', 'subjectivity']].head())

plt.hist(df['polarity'], bins=20, color='blue', edgecolor='black')
plt.title('Rozkład polaryzacji sentymentu')
plt.xlabel('Polaryzacja')
plt.ylabel('Liczba recenzji')
plt.show()
