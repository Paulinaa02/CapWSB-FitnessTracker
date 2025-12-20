"""
API Reference - SimpleVectorizer
=================================

Kompletna dokumentacja interfejsu klasy SimpleVectorizer.
"""

# ============================================================================
# CLASS: SimpleTokenizer
# ============================================================================

class SimpleTokenizer:
    """
    Klasa do tokenizacji i preprocesingu tekstu.
    
    Parametry:
    ----------
    lowercase : bool, default=True
        Jeśli True, konwertuje tekst na małe litery przed tokenizacją.
    
    use_stemming : bool, default=False
        Jeśli True, stosuje Porter Stemming do każdego tokenu.
        Wymaga biblioteki nltk (opcjonalnej).
    
    remove_diacritics : bool, default=False
        Jeśli True, usuwa znaki diakrytyczne (np. ą->a, ć->c).
    
    Metody:
    -------
    tokenize(text, ngram_range=(1, 1)) -> List[str]
        Tokenizuje tekst na słowa/n-gramy.
        
        Parametry:
            text (str): Tekst do tokenizacji
            ngram_range (tuple): (min_n, max_n) - zakres n-gramów
        
        Zwraca:
            Lista tokenów
        
        Przykład:
            >>> tokenizer = SimpleTokenizer(lowercase=True)
            >>> tokenizer.tokenize("Hello World!")
            ['hello', 'world']
            
            >>> tokenizer.tokenize("Hello World", ngram_range=(1, 2))
            ['hello', 'world', 'hello world']
    
    remove_accents(text) -> str
        Usuwa znaki diakrytyczne z tekstu.
        
        Parametry:
            text (str): Tekst do przetworzenia
        
        Zwraca:
            Tekst bez znaków diakrytycznych
        
        Przykład:
            >>> tokenizer.remove_accents("café")
            'cafe'
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 use_stemming: bool = False,
                 remove_diacritics: bool = False):
        pass
    
    def tokenize(self, text: str, ngram_range: tuple = (1, 1)) -> list:
        """Tokenizacja tekstu"""
        pass
    
    def remove_accents(self, text: str) -> str:
        """Usuwanie znaków diakrytycznych"""
        pass


# ============================================================================
# CLASS: SimpleVectorizer
# ============================================================================

class SimpleVectorizer:
    """
    Wektoryzator Bag-of-Words i TF-IDF od podstaw.
    
    Implementuje interfejs fit/transform podobny do scikit-learn.
    
    Parametry:
    ----------
    lowercase : bool, default=True
        Konwersja tekstu na małe litery.
    
    stop_words : list or 'english', optional
        Lista słów stopowych do usunięcia.
        - 'english': Używa wbudowanej angielskiej stoplisty
        - list: Custom lista słów
        - None: Bez stoplisity
    
    min_df : int or float, default=1
        Ignoruj tokeny pojawiające się w mniej niż min_df dokumentach.
        - Jeśli int: bezwzględna liczba dokumentów
        - Jeśli float (0-1): odsetek dokumentów
        
        Przykład:
            min_df=2: Ignoruj tokeny w <2 dokumentach
            min_df=0.05: Ignoruj tokeny w <5% dokumentów
    
    max_df : int or float, default=1.0
        Ignoruj tokeny pojawiające się w więcej niż max_df dokumentach.
        - Jeśli int: bezwzględna liczba dokumentów
        - Jeśli float (0-1): odsetek dokumentów
        
        Przykład:
            max_df=100: Ignoruj tokeny w >100 dokumentach
            max_df=0.8: Ignoruj tokeny w >80% dokumentów
    
    max_features : int, optional
        Maksymalna liczba features (tokenów) w słowniku.
        Używa najczęstszych tokenów.
        
        Przykład:
            max_features=1000: Słownik zawiera maks. 1000 tokenów
    
    binary : bool, default=False
        Jeśli True, BoW zawiera 0/1 (obecność).
        Jeśli False, BoW zawiera liczby (count).
        
        Przykład:
            binary=False: [0, 2, 1, 3] (liczby)
            binary=True:  [0, 1, 1, 1] (binarne)
    
    use_idf : bool, default=True
        Jeśli True, używa TF-IDF zamiast czystego BoW.
    
    smooth_idf : bool, default=True
        Jeśli True, dodaje 1 do licznika i mianownika IDF:
            IDF = log((N+1)/(n_t+1)) + 1
        Jeśli False:
            IDF = log(N/n_t)
    
    sublinear_tf : bool, default=False
        Jeśli True, stosuje sublinear scaling dla TF:
            TF = 1 + log(TF)
        Zmniejsza wpływ bardzo częstych słów.
    
    norm : {None, 'l1', 'l2'}, default=None
        Normalizacja wektorów:
        - None: Bez normalizacji
        - 'l1': Normalizacja L1 (Manhattan distance)
        - 'l2': Normalizacja L2 (Euclidean distance, domyślnie dla TF-IDF)
    
    handle_oov : {'ignore', 'add_column', 'error'}, default='ignore'
        Obsługa tokenów Out-Of-Vocabulary (nie w słowniku):
        - 'ignore': Pomiń tokeny nie w słowniku
        - 'add_column': Dodaj kolumnę '<OOV>' z liczbą nieznanych tokenów
        - 'error': Wyświetl ostrzeżenie
    
    use_stemming : bool, default=False
        Jeśli True, aplikuje Porter Stemming. Wymaga nltk.
    
    ngram_range : tuple of (min_n, max_n), default=(1, 1)
        Zakres n-gramów.
        - (1, 1): Tylko unigramy
        - (2, 2): Tylko bigramy
        - (1, 2): Unigramy + bigramy
    
    remove_diacritics : bool, default=False
        Usuwaj znaki diakrytyczne z tekstu.
    
    Atrybuty (dostępne po fit()):
    -----------------------------
    vocabulary_ : dict
        Słownik {token -> indeks kolumny}
        Mapuje tokeny do ich indeksów w wektorach.
    
    idf_ : numpy.ndarray, shape (n_features,)
        Wartości IDF dla każdego tokenu.
        Dostęp: idf_[vocabulary_['token']]
    
    document_frequencies_ : dict
        Liczba dokumentów zawierających każdy token.
    
    n_docs_fit_ : int
        Liczba dokumentów w korpusie treningowym.
    
    is_fitted : bool
        Czy vectorizer został już trenowany.
    
    Metody:
    -------
    fit(X) -> SimpleVectorizer
        Trenuje vectorizer na korpusie.
        
        Parametry:
            X (List[str]): Lista dokumentów (łańcuchów znakowych)
        
        Zwraca:
            self (dla łańcuchowania)
        
        Przykład:
            >>> corpus = ["Hello world", "Python is fun"]
            >>> vec = SimpleVectorizer()
            >>> vec.fit(corpus)
            >>> vec.vocabulary_
            {'hello': 0, 'world': 1, 'python': 2, ...}
    
    transform(X, return_df=False) -> numpy.ndarray or pandas.DataFrame
        Konwertuje dokumenty na wektory.
        
        Parametry:
            X (List[str]): Lista dokumentów do transformacji
            return_df (bool): Zwróć DataFrame zamiast array?
        
        Zwraca:
            - numpy.ndarray shape (n_docs, n_features) jeśli return_df=False
            - pandas.DataFrame jeśli return_df=True
        
        Podnosi:
            ValueError: Jeśli vectorizer nie został fit()
        
        Przykład:
            >>> X_test = ["hello python"]
            >>> vectors = vec.transform(X_test, return_df=True)
            >>> print(vectors)
                hello  world  python  is  fun
            0      1      0       1   0    0
    
    fit_transform(X, return_df=False) -> numpy.ndarray or pandas.DataFrame
        fit() i transform() w jednym kroku (bardziej efektywne).
        
        Parametry:
            X (List[str]): Lista dokumentów
            return_df (bool): Zwróć DataFrame?
        
        Zwraca:
            Macierz wektorów
        
        Przykład:
            >>> corpus = ["Hello world", "Python is fun"]
            >>> vec = SimpleVectorizer()
            >>> X = vec.fit_transform(corpus)
            >>> X.shape
            (2, 4)
    
    get_feature_names_out() -> numpy.ndarray
        Zwraca nazwy features (tokeny) w kolejności indeksów.
        
        Zwraca:
            Array tokenów w kolejności: ['token0', 'token1', ...]
        
        Przykład:
            >>> vec.get_feature_names_out()
            array(['hello', 'world', 'python', 'is', 'fun'], dtype=object)
    
    save(path) -> None
        Zapisuje stan vectorizatora do pliku.
        
        Parametry:
            path (str): Ścieżka do pliku (np. 'model.pkl')
        
        Przykład:
            >>> vec.save('my_vectorizer.pkl')
    
    load(path) -> SimpleVectorizer
        Ładuje stan vectorizatora z pliku.
        
        Parametry:
            path (str): Ścieżka do pliku
        
        Zwraca:
            self (dla łańcuchowania)
        
        Przykład:
            >>> vec = SimpleVectorizer()
            >>> vec.load('my_vectorizer.pkl')
            >>> vec.transform(new_documents)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Przykład 1: Prostych BoW
"""
from vectorizer import SimpleVectorizer

corpus = [
    "Python programming is fun",
    "Machine learning with Python",
    "Data science and Python"
]

vec = SimpleVectorizer(
    binary=False,      # Liczby, nie binarne
    use_idf=False,     # Czysty BoW, bez TF-IDF
    lowercase=True
)

# Trenowanie i transformacja
matrix = vec.fit_transform(corpus, return_df=True)
print("BoW Matrix:")
print(matrix)
print("\nVocabulary:", vec.vocabulary_)


"""
Przykład 2: TF-IDF z normalizacją L2
"""
vec_tfidf = SimpleVectorizer(
    use_idf=True,
    smooth_idf=True,
    norm='l2'           # Normalizacja L2
)

X_tfidf = vec_tfidf.fit_transform(corpus, return_df=True)
print("\nTF-IDF Matrix (L2-normalized):")
print(X_tfidf.round(3))


"""
Przykład 3: Obsługa OOV i nowe dokumenty
"""
vec = SimpleVectorizer(handle_oov='add_column')
vec.fit(corpus)

new_docs = [
    "Python is awesome",    # 'awesome' - nowe słowo
    "Deep learning"         # 'deep' - nowe słowo
]

X_new = vec.transform(new_docs, return_df=True)
print("\nOOV Handling:")
print(X_new)
# Kolumna '<OOV>' zawiera liczbę nieznanych słów


"""
Przykład 4: Stoplista i filtrowanie
"""
vec = SimpleVectorizer(
    stop_words='english',   # Wbudowana angielska stoplista
    min_df=2,               # Ignoruj słowa w <2 dokumentach
    max_features=50         # Maks 50 tokenów
)

vec.fit(corpus)
print("\nFiltered vocabulary size:", len(vec.vocabulary_))


"""
Przykład 5: N-gramy i stemming
"""
vec = SimpleVectorizer(
    ngram_range=(1, 2),     # Unigramy + bigramy
    use_stemming=True       # Porter Stemming
)

vec.fit(corpus)
print("\nFeatures with bigrams:")
print(vec.get_feature_names_out())


# ============================================================================
# PORÓWNANIE Z SCIKIT-LEARN
# ============================================================================

"""
Interfejs jest kompatybilny ze scikit-learn!

# Nasze SimpleVectorizer
from vectorizer import SimpleVectorizer
vec = SimpleVectorizer(use_idf=True, norm='l2')
X = vec.fit_transform(corpus)

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(norm='l2')
X = vec.fit_transform(corpus)

# Wyniki są zbliżone (mogą być małe różnice w algorytmach)
"""


# ============================================================================
# WSKAZÓWKI I BEST PRACTICES
# ============================================================================

"""
1. Wybieranie parametrów:
   - Dla klasyfikacji: use_idf=True, norm='l2'
   - Dla wyszukiwania: binary=True (szybciej)
   - Dla tekstu o wielu językach: remove_diacritics=True
   - Dla tekstu naturalnego: stop_words='english', min_df=2

2. Obsługa dużych zbiorów:
   - max_features=5000 (ograniczy rozmiar słownika)
   - min_df >= 2 (usunie rzadkie słowa)
   - max_df < 0.9 (usunie bardzo częste słowa)

3. Debugowanie:
   - Sprawdzaj rozmiar vocabulary_: len(vec.vocabulary_)
   - Analizuj IDF: np.argsort(-vec.idf_) daje najważniejsze słowa
   - Sprawdzaj rozmiar macierzy: X.shape

4. Optymalizacje:
   - Używaj return_df=False dla NumPy arrays (szybciej)
   - Zapisuj model z save() po trenowaniu
   - Używaj fit_transform zamiast fit().transform()

5. Produkcja:
   - Zawsze zachowaj vectorizer z save()
   - Używaj tego samego vectorizatora do transform
   - Obsługuj OOV z handle_oov='add_column'
"""