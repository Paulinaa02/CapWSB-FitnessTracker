# SimpleVectorizer - Bag-of-Words i TF-IDF od Podstaw

Implementacja wektoryzatorów tekstowych bez użycia scikit-learn.

## 📋 Spis Treści

1. [Instalacja](#instalacja)
2. [Szybki Start](#szybki-start)
3. [Dokumentacja](#dokumentacja)
4. [Przykłady](#przykłady)
5. [Architektura](#architektura)
6. [Testy](#testy)

---

## Instalacja

### Wymagania

- Python 3.8+
- NumPy
- Pandas
- SciPy (opcjonalnie)
- NLTK (opcjonalnie, dla stemmingu)

### Pobranie

```bash
# Skopiuj plik vectorizer.py do swojego projektu
cp vectorizer.py /twoj/projekt/

# Lub bezpośrednio z kodu
from vectorizer import SimpleVectorizer, SimpleTokenizer
```

### Instalacja Zależności

```bash
pip install numpy pandas scipy
pip install nltk  # Opcjonalnie, dla stemmingu
```

---

## Szybki Start

### Bag-of-Words (BoW)

```python
from vectorizer import SimpleVectorizer

# Dokumenty
corpus = [
    "Python jest wspaniałym językiem",
    "Machine learning to przyszłość",
    "Python i machine learning"
]

# Utwórz i trenuj vectorizer
vec = SimpleVectorizer(binary=False, use_idf=False)
matrix = vec.fit_transform(corpus, return_df=True)

print(matrix)
```

**Output:**
```
   Python  jest  wspaniałym  ...
0      1     1           1  ...
1      0     0           0  ...
2      1     0           0  ...
```

### TF-IDF

```python
# TF-IDF z normalizacją L2
vec = SimpleVectorizer(use_idf=True, norm='l2')
tfidf_matrix = vec.fit_transform(corpus, return_df=True)

print(tfidf_matrix.round(3))
```

---

## Dokumentacja

### Parametry SimpleVectorizer

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|----------|------|
| `lowercase` | bool | True | Konwersja na małe litery |
| `stop_words` | list/str | None | Stoplista ('english' lub custom) |
| `min_df` | int/float | 1 | Min. liczba dokumentów dla tokenu |
| `max_df` | int/float | 1.0 | Max. liczba dokumentów dla tokenu |
| `max_features` | int | None | Max. liczba tokenów |
| `binary` | bool | False | Zwracaj 0/1 zamiast liczb |
| `use_idf` | bool | True | Oblicz TF-IDF |
| `smooth_idf` | bool | True | Smoothing dla IDF |
| `sublinear_tf` | bool | False | Sublinearne skalowanie TF |
| `norm` | str | None | Normalizacja ('l1', 'l2') |
| `handle_oov` | str | 'ignore' | Obsługa nieznanych tokenów |
| `use_stemming` | bool | False | Porter Stemming |
| `ngram_range` | tuple | (1,1) | Zakres n-gramów |
| `remove_diacritics` | bool | False | Usuwaj znaki diakrytyczne |

### Główne Metody

```python
# Trenowanie
vec.fit(documents) -> SimpleVectorizer

# Transformacja
vec.transform(documents, return_df=False) -> np.ndarray / DataFrame

# Fit + Transform
vec.fit_transform(documents, return_df=False) -> np.ndarray / DataFrame

# Pobierz nazwy tokenów
vec.get_feature_names_out() -> np.ndarray

# Zapis modelu
vec.save(path)

# Ładowanie modelu
vec.load(path)
```

---

## Przykłady

### Przykład 1: Prosty BoW

```python
from vectorizer import SimpleVectorizer

corpus = ["hello world", "hello python", "world of python"]

vec = SimpleVectorizer()
X = vec.fit_transform(corpus, return_df=True)

print("Słownik:", vec.vocabulary_)
# {'hello': 0, 'world': 1, 'python': 2, 'of': 3}

print("\nWektory:")
print(X)
```

### Przykład 2: TF-IDF z Normalizacją L2

```python
vec = SimpleVectorizer(
    use_idf=True,
    smooth_idf=True,
    norm='l2'
)

X = vec.fit_transform(corpus, return_df=True)
print("TF-IDF (znormalizowany):")
print(X.round(3))

# Sprawdź normy (powinny być ~1.0)
import numpy as np
for i in range(X.shape[0]):
    norm = np.linalg.norm(X.iloc[i])
    print(f"Dokument {i}: L2 norm = {norm:.4f}")
```

### Przykład 3: Obsługa Stoplisity

```python
# Wbudowana angielska stoplista
vec = SimpleVectorizer(stop_words='english')
vec.fit(corpus)

# Custom stoplista
custom_stops = ['and', 'the', 'is']
vec = SimpleVectorizer(stop_words=custom_stops)
vec.fit(corpus)

print(f"Liczba tokenów: {len(vec.vocabulary_)}")
```

### Przykład 4: N-gramy

```python
# Tylko bigramy (pary słów)
vec_bigrams = SimpleVectorizer(ngram_range=(2, 2))
vec_bigrams.fit(corpus)

# Unigramy + bigramy
vec_both = SimpleVectorizer(ngram_range=(1, 2))
vec_both.fit(corpus)

print("Features:", vec_both.get_feature_names_out())
# ['hello', 'world', 'python', 'of', 'hello world', 'world python', ...]
```

### Przykład 5: Obsługa OOV (Out-Of-Vocabulary)

```python
# Trenuj na jednym zbiorze
train_docs = ["hello world", "python programming"]

# Test na dokumentach z nowymi słowami
test_docs = ["hello universe", "java programming"]  # 'universe' i 'java' - nowe

vec = SimpleVectorizer(handle_oov='add_column')
vec.fit(train_docs)

X_test = vec.transform(test_docs, return_df=True)
print(X_test)
# Kolumna '<OOV>' zawiera liczbę nieznanych słów w każdym dokumencie
```

### Przykład 6: Filtrowanie min_df i max_df

```python
# Ignoruj słowa w mniej niż 2 dokumentach
vec = SimpleVectorizer(min_df=2)
vec.fit(corpus)

# Ignoruj słowa w więcej niż 80% dokumentów
vec = SimpleVectorizer(max_df=0.8)
vec.fit(corpus)

# Kombinacja
vec = SimpleVectorizer(min_df=2, max_df=0.8, max_features=1000)
vec.fit(corpus)
```

### Przykład 7: Zapis i Ładowanie Modelu

```python
# Trenuj model
vec = SimpleVectorizer(use_idf=True, norm='l2')
vec.fit(corpus)

# Zapisz
vec.save('my_vectorizer.pkl')

# Załaduj do transformacji nowych dokumentów
vec_loaded = SimpleVectorizer()
vec_loaded.load('my_vectorizer.pkl')

new_X = vec_loaded.transform(new_documents)
```

---

## Architektura

### Klasy

#### `SimpleTokenizer`

Odpowiadialność: Tokenizacja i preprocesing tekstu

```python
tokenizer = SimpleTokenizer(
    lowercase=True,           # Konwersja na małe litery
    use_stemming=False,       # Porter Stemming
    remove_diacritics=False   # Usuwanie znaków diakrytycznych
)

tokens = tokenizer.tokenize(
    "Hello World!",
    ngram_range=(1, 1)  # (1,1) = unigramy, (1,2) = uni+bigrams
)
# Zwraca: ['hello', 'world']
```

#### `SimpleVectorizer`

Odpowiedzialność: Vectoryzacja tekstów (BoW i TF-IDF)

**Workflow:**
1. **fit(X)** - Buduje słownik z korpusu
2. **transform(X)** - Konwertuje dokumenty na wektory
3. **fit_transform(X)** - Oba kroki naraz

```python
vec = SimpleVectorizer(
    use_idf=True,
    norm='l2',
    stop_words='english',
    min_df=2,
    max_df=0.8
)

# Trenowanie
vec.fit(train_docs)

# Transformacja
X_train = vec.transform(train_docs)
X_test = vec.transform(test_docs)

# Atrybuty po fit
print(vec.vocabulary_)           # {token -> indeks}
print(vec.idf_)                  # [wartości IDF dla każdego tokenu]
print(vec.document_frequencies_) # {token -> liczba dokumentów}
```

### Matematyka

**Bag-of-Words:**
$$\text{BoW}(t, d) = \text{count}(t \text{ in } d)$$

**TF-IDF:**
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{n_t}\right)$$

Gdzie:
- $t$ = token (słowo)
- $d$ = dokument
- $N$ = całkowita liczba dokumentów
- $n_t$ = liczba dokumentów zawierających token $t$

---

## Testy

### Uruchomienie Testów

Notebook `Vectorizer_BoW_TF-IDF.ipynb` zawiera 7 testów jednostkowych:

```python
# W notebooku -> Komórka "Testy jednostkowe"

✓ test_basic_vocabulary()        # Słownik
✓ test_transform_shape()         # Wymiary macierzy
✓ test_bow_counts()              # Liczenia BoW
✓ test_binary_bow()              # Binary wartości
✓ test_l2_normalization()        # L2 normalizacja
✓ test_min_df_filtering()        # Filtrowanie min_df
✓ test_oov_handling()            # Obsługa OOV
```

### Własne Testy

```python
def test_custom():
    corpus = ["a b c", "b c d"]
    vec = SimpleVectorizer()
    X = vec.fit_transform(corpus, return_df=True)
    
    assert X.shape == (2, 4), f"Expected (2,4), got {X.shape}"
    assert vec.vocabulary_['a'] == 0
    assert X.loc[0, 'a'] == 1
    print("✓ Test passed")

test_custom()
```

---

## Porównanie z scikit-learn

### Kompatybilność Interfejsu

```python
# Nasz kod
from vectorizer import SimpleVectorizer
vec = SimpleVectorizer(use_idf=True, norm='l2')
X = vec.fit_transform(corpus)

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
vec_sk = TfidfVectorizer(norm='l2')
X_sk = vec_sk.fit_transform(corpus)

# Wyniki są zbliżone
import numpy as np
similarity = np.allclose(X.toarray(), X_sk.toarray())
```

### Różnice

| Cecha | SimpleVectorizer | scikit-learn |
|-------|------------------|--------------|
| Sparse arrays | Nie | Tak |
| Wydajność | Dla małych zbiorów | Bardzo szybka |
| Łatwość modyfikacji | Wysoka | Niska |
| Dokumentacja | Właśnie czytasz | Obszernie |
| Online learning | Nie | Nie bezpośrednio |

---

## Optymalizacje i Rozszerzenia

### Zastosowane

- ✓ Vectorizacja NumPy
- ✓ Counter dla efektywnego liczenia
- ✓ Lazy IDF computation
- ✓ Copy-on-write optymalizacje

### Przyszłe Ulepszenia

- [ ] Sparse matrices (scipy.sparse)
- [ ] Incremental fit (online learning)
- [ ] Parallelizacja
- [ ] Custom tokenizers
- [ ] Caching

---

## Licencja

Projekt edukacyjny - public domain

## Autor

Implementacja własna, 2025

## Pliki Projektu

```
CapWSB-FitnessTracker/
├── vectorizer.py                    # Implementacja modułu
├── Vectorizer_BoW_TF-IDF.ipynb     # Notebook z przykładami
├── SPRAWOZDANIE.md                 # Dokumentacja projektu
├── API_REFERENCE.py                # Dokumentacja API
└── README.md                       # Ten plik
```

---

## Szybkie Rozwiązywanie Problemów

### Problem: "Vectorizer not fitted"

```python
vec = SimpleVectorizer()
X = vec.transform(docs)  # ❌ Błąd!

# Rozwiązanie
vec.fit(docs)            # Najpierw fit
X = vec.transform(docs)  # Potem transform
```

### Problem: Za wiele tokenów

```python
vec = SimpleVectorizer(max_features=1000)  # Limit na 1000
vec = SimpleVectorizer(min_df=2)           # Min. w 2 dokumentach
vec = SimpleVectorizer(max_df=0.9)         # Max. w 90% dokumentów
```

### Problem: Nowe słowa w transform

```python
vec = SimpleVectorizer(handle_oov='add_column')
# Nowe słowa będą w kolumnie '<OOV>'
```

### Problem: Wolne działanie

```python
# Użyj return_df=False (szybciej)
X = vec.transform(docs, return_df=False)

# Limit tokenów
vec = SimpleVectorizer(max_features=5000)

# Filtrowanie
vec = SimpleVectorizer(min_df=2, max_df=0.9)
```

---

**Powodzenia w pracy z SimpleVectorizer!** 🚀