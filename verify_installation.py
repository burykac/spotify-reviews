def verify():
    try:
        import pandas as pd
        import numpy as np
        import nltk
        import spacy
        from sklearn.feature_extraction.text import TfidfVectorizer
        from textblob import TextBlob
        from kaggle.api.kaggle_api_extended import KaggleApi

        print("Wszystkie biblioteki zostały poprawnie zaimportowane.")
    except ImportError as e:
        print(f"ImportError: {e}")