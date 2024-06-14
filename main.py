import os
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

verify()
check_kaggle_api()
download()

csv_file_path = os.path.join('data', 'spotify_reviews.csv')

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print(df.head())
else:
    print(f"Plik {csv_file_path} nie istnieje.")

