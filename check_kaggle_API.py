from kaggle.api.kaggle_api_extended import KaggleApi

def check_kaggle_api():
    try:
        api = KaggleApi()
        api.authenticate()
        print("API Kaggle zostało poprawnie skonfigurowane i uwierzytelnione.")
    except Exception as e:
        print(f"Błąd uwierzytelnienia: {e}")

check_kaggle_api()