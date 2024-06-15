Analiza recenzji Spotify z Google Play Store

Opis projektu

Ten projekt analizuje recenzje aplikacji Spotify pobrane z Google Play Store. Główne cele projektu to:

Pobranie i weryfikacja danych z Kaggle.
Wstępna analiza danych, w tym oceny aplikacji.
Przetwarzanie i czyszczenie tekstu recenzji.
Wektoryzacja tekstu za pomocą TF-IDF.
Analiza sentymentu recenzji.
Struktura projektu
Projekt składa się z kilku plików i katalogów:

main.py: Główny plik wykonywalny projektu.
check_kaggle_API.py: Skrypt sprawdzający konfigurację i uwierzytelnienie API Kaggle.
download_data.py: Skrypt pobierający dane z Kaggle.
verify_installation.py: Skrypt sprawdzający, czy wszystkie wymagane biblioteki są poprawnie zainstalowane.
requirements.txt: Plik z listą wszystkich wymaganych pakietów.
data/: Katalog przechowujący pobrane dane.


Instalacja
Aby uruchomić projekt, wykonaj poniższe kroki:

Sklonuj repozytorium:

git clone <https://github.com/burykac/spotify-reviews.git>

cd <spotify-reviews>


Zainstaluj wymagane pakiety:

pip install -r requirements.txt


Skonfiguruj API Kaggle:

Utwórz plik ~/.kaggle/kaggle.json z danymi uwierzytelniającymi API Kaggle. Plik powinien zawierać klucz API w formacie JSON:
json

{
  "username": "YOUR_USERNAME",
  "key": "YOUR_KEY"
}

Uruchomienie
Sprawdź instalację pakietów:

python verify_installation.py

Pobierz dane z Kaggle:

python download_data.py

Uruchom główny skrypt:

python main.py

Opis funkcji i klas

main.py

Importowanie niezbędnych pakietów.
Pobieranie i weryfikacja danych.
Wstępna analiza danych, w tym obliczanie średniej oceny.
Klasa DataLoader do przechowywania DataFrame.
Klasa TextProcessor do przetwarzania tekstu recenzji:
Normalizacja tekstu.
Usuwanie znaków przestankowych.
Usuwanie znaków specjalnych.
Tokenizacja tekstu.
Usuwanie stopwords.
Stemmatyzacja i lematyzacja tokenów.
Klasa Vectorizer do wektoryzacji tekstu za pomocą TF-IDF.
Analiza sentymentu za pomocą TextBlob.

check_kaggle_API.py

Funkcja check_kaggle_api sprawdzająca konfigurację i uwierzytelnienie API Kaggle.
download_data.py
Funkcja download_kaggle_dataset pobierająca dane z Kaggle.
Funkcja download zarządzająca procesem pobierania danych.

verify_installation.py

Funkcja verify sprawdzająca, czy wszystkie wymagane biblioteki są poprawnie zainstalowane.

Wizualizacje
Projekt zawiera wizualizacje danych:

Histogram ocen aplikacji.
Histogram polaryzacji sentymentu recenzji.

Wyniki
Po przetworzeniu danych i analizie sentymentu, wyniki są wyświetlane na konsoli oraz wizualizowane za pomocą matplotlib.

Kontakt
Jeśli masz pytania dotyczące projektu, proszę o kontakt na adres e-mail: kbury@edu.cdv.pl
