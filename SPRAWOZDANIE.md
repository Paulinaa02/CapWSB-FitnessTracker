# Sprawozdanie: Implementacja Wektoryzatorów BoW i TF-IDF

## 1. Wstęp

Projekt obejmuje implementację klasy `SimpleVectorizer`, która umożliwia konwertowanie dokumentów tekstowych na wektory numeryczne w reprezentacjach Bag-of-Words (BoW) i TF-IDF. Implementacja została wykonana bez użycia bibliotek `CountVectorizer` i `TfidfVectorizer` ze scikit-learn.

---

## 2. Architektura Rozwiązania

### 2.1 Klasa `SimpleTokenizer`

Odpowiedzialność: Tokenizacja i preprocesing tekstu

**Funkcjonalności:**
- Konwersja tekstu na małe litery (parametr `lowercase`)
- Tokenizacja za pomocą wyrażeń regularnych: `\w+` (słowa, liczby, podkreślenia)
- Opcjonalny stemming tekstu za pomocą Porter Stemmera
- Usuwanie znaków diakrytycznych
- Obsługa n-gramów (unigramy, bigramy, trigramy, etc.)

**Przykład użycia:**
```python
tokenizer = SimpleTokenizer(lowercase=True, use_stemming=False)
tokens = tokenizer.tokenize("Hello World!", ngram_range=(1, 2))
# Zwraca: ['hello', 'world', 'hello world']
```

### 2.2 Klasa `SimpleVectorizer`

Główna klasa implementująca interfejs fit/transform zgodny ze scikit-learn.

**Metoda `fit(X: List[str])`**
- Buduje słownik tokenów z podanego korpusu dokumentów
- Liczy częstość dokumentów dla każdego tokenu (`document_frequencies_`)
- Oblicza wartości IDF dla wszystkich tokenów
- Filtruje tokeny według kryteriów (`min_df`, `max_df`, `max_features`)
- Zwraca instancję `self` dla łańcuchowania metod

**Metoda `transform(X: List[str]) -> np.ndarray`**
- Konwertuje dokumenty na wektory numeryczne
- Obsługuje reprezentacje: BoW (liczby/binarne) i TF-IDF
- Obsługuje sublinear TF scaling
- Normaliza wektory (L1, L2)
- Obsługuje tokeny Out-Of-Vocabulary (OOV)
- Zwraca macierz NumPy lub DataFrame Pandas

**Metoda `fit_transform(X: List[str])`**
- Kombinacja fit i transform w jednym kroku
- Bardziej efektywna niż oddzielne wywołania

---

## 3. Kluczowe Parametry Vectorizatora

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|----------|------|
| `lowercase` | bool | True | Konwersja tekstu na małe litery |
| `stop_words` | list/str | None | Stoplista do filtrowania ('english' lub custom) |
| `min_df` | int/float | 1 | Minimalna liczba/% dokumentów dla tokenu |
| `max_df` | int/float | 1.0 | Maksymalna liczba/% dokumentów dla tokenu |
| `max_features` | int | None | Limit na liczbę tokenów w słowniku |
| `binary` | bool | False | Zwracaj 0/1 zamiast liczb (BoW) |
| `use_idf` | bool | True | Oblicz TF-IDF zamiast czystego BoW |
| `smooth_idf` | bool | True | Smoothing dla IDF (unika dzielenia przez 0) |
| `sublinear_tf` | bool | False | Sublinearne skalowanie TF: $1 + \log(tf)$ |
| `norm` | str | None | Normalizacja wektorów: 'l1', 'l2', None |
| `handle_oov` | str | 'ignore' | Obsługa nieznanych tokenów |
| `use_stemming` | bool | False | Zastosuj stemming (Porter) |
| `ngram_range` | tuple | (1,1) | Zakres n-gramów |

---

## 4. Matematyka

### 4.1 Bag-of-Words (BoW)

Najprostsza reprezentacja: liczba wystąpień każdego słowa w dokumencie.

$$\text{BoW}(t, d) = \text{count}(t \in d)$$

Gdzie:
- $t$ = token (słowo)
- $d$ = dokument

**Wariant binarny:**
$$\text{BoW}_{\text{binary}}(t, d) = \begin{cases} 1 & \text{jeśli } t \in d \\ 0 & \text{w p.p.} \end{cases}$$

### 4.2 Term Frequency (TF)

Znormalizowana liczba występień:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}}$$

Gdzie:
- $f_{t,d}$ = liczba razy token $t$ pojawia się w $d$
- Suma w mianowniku = całkowita liczba tokenów w $d$

**Sublinearne skalowanie:**
$$\text{TF}_{\text{sublinear}}(t, d) = 1 + \log(f_{t,d})$$

### 4.3 Inverse Document Frequency (IDF)

Mierzy ważność tokenu w całym korpusie:

$$\text{IDF}(t) = \log\left(\frac{N}{n_t}\right)$$

**Ze smoothingiem (domyślnie):**
$$\text{IDF}(t) = \log\left(\frac{N + 1}{n_t + 1}\right) + 1$$

Gdzie:
- $N$ = całkowita liczba dokumentów
- $n_t$ = liczba dokumentów zawierających token $t$

### 4.4 TF-IDF

Kombinacja TF i IDF:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Tokenów występujące w wielu dokumentach (niska IDF) mają mniejszy wpływ, natomiast tokeny rzadkie (wysoka IDF) są bardziej ważne.

### 4.5 Normalizacja wektorów

**Normalizacja L2 (domyślnie dla TF-IDF):**
$$v' = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_i v_i^2}}$$

Skaluje wektory do jednostkowej długości.

**Normalizacja L1 (Manhattan):**
$$v' = \frac{v}{\|v\|_1} = \frac{v}{\sum_i |v_i|}$$

Normalizuje sumę komponentów do 1.

---

## 5. Obsługa Out-Of-Vocabulary (OOV) Tokenów

Podczas `transform` mogą pojawić się tokeny, które nie były w zbiorze treningowym podczas `fit`.

**Strategie:**
1. **'ignore'** (domyślnie): Pomiń nieznane tokeny w transformacji
2. **'add_column'**: Dodaj kolumnę `<OOV>` zliczającą nieznane tokeny
3. **'error'**: Wyświetl ostrzeżenie

**Przykład:**
```python
vec = SimpleVectorizer(handle_oov='add_column')
vec.fit(["hello world"])
# Transform dokument z nowym słowem
result = vec.transform(["hello universe"])
# Kolumna '<OOV>' zawiera 1 (dla 'universe')
```

---

## 6. Filtry Dokumentów

### 6.1 `min_df` - Minimalna Czestość

Ignoruj tokeny pojawiające się w mniej niż `min_df` dokumentach.

```python
vec = SimpleVectorizer(min_df=2)
# Tokeny, które pojawiają się w <2 dokumentach, będą usunięte
```

### 6.2 `max_df` - Maksymalna Czestość

Ignoruj tokeny pojawiające się w więcej niż `max_df` dokumentach.

```python
vec = SimpleVectorizer(max_df=0.8)
# Tokeny w >80% dokumentów będą usunięte
# Przydatne do usuwania "stopwords" bez konkretnej listy
```

### 6.3 `max_features` - Limit Liczby Tokenów

Zachowaj tylko `max_features` najczęstszych tokenów.

```python
vec = SimpleVectorizer(max_features=1000)
# Słownik będzie zawierać maksymalnie 1000 tokenów
```

---

## 7. N-gramy

Zamiast pojedynczych słów (unigramów), wektoryzator może pracować z sekwencjami słów.

```python
vec_bigrams = SimpleVectorizer(ngram_range=(2, 2))
# Токены: "machine learning", "learning is", "is fun"

vec_mixed = SimpleVectorizer(ngram_range=(1, 2))
# Tokeny: unigramy + bigramy
```

**Użyteczność:**
- Bigramy mogą uchwycić kontekst (np. "machine learning" vs "learning machine")
- Lepsze przy analizie sentymentu ("not good" vs "good")

---

## 8. Stoplista (Stop Words)

Wyrazy, które są mało informatywne i mogą być usunięte z analizy.

```python
# Wbudowana angielska stoplista
vec = SimpleVectorizer(stop_words='english')

# Custom stoplista
custom_stops = ['and', 'or', 'the', 'is']
vec = SimpleVectorizer(stop_words=custom_stops)
```

**Przykłady angielskich stopwords:**
"a", "an", "the", "and", "or", "is", "was", "be", "have", etc.

---

## 9. Porównanie z scikit-learn

### Cechy Naszej Implementacji

**Zalety:**
✓ Łatwa do zrozumienia i modyfikacji  
✓ Pełna kontrola nad procesem  
✓ Edukacyjna wartość  
✓ Nie wymaga dodatkowych zależności (poza NumPy, Pandas)

**Ograniczenia:**
✗ Mniejsza wydajność na dużych zbiorach  
✗ Nie zawiera sparse matrices (scipy.sparse)  
✗ Brak incremental fit (online learning)

### Porównanie Interfejsu

```python
# Nasza implementacja
from vectorizer import SimpleVectorizer
vec = SimpleVectorizer(use_idf=True, norm='l2')
X = vec.fit_transform(corpus)

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(norm='l2')
X = vec.fit_transform(corpus)
```

Interfejsy są kompatybilne!

---

## 10. Praktyczne Zastosowania

### 10.1 Klasyfikacja Tekstu

```python
# Wektoryzacja tekstu dla klasyfikatora
vec = SimpleVectorizer(use_idf=True, norm='l2', stop_words='english')
X_train = vec.fit_transform(train_docs)
X_test = vec.transform(test_docs)

# Trenowanie klasyfikatora (np. Naive Bayes)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### 10.2 Wyszukiwanie Tekstowe (Information Retrieval)

```python
# Budowanie indeksu
vec = SimpleVectorizer(binary=True)  # Binary BoW szybsza
corpus_vectors = vec.fit_transform(corpus)

# Query
query_vector = vec.transform([query])

# Wyszukiwanie podobnych dokumentów
similarities = corpus_vectors.dot(query_vector.T)
top_docs = np.argsort(-similarities.ravel())[:10]
```

### 10.3 Analiza Dokumentów

```python
# Znalezienie najważniejszych słów
vec = SimpleVectorizer(use_idf=True)
vec.fit(documents)

# IDF - słowa charakterystyczne dla korpusu
important_words = np.argsort(-vec.idf_)[:20]
for idx in important_words:
    token = vec.get_feature_names_out()[idx]
    idf = vec.idf_[idx]
    print(f"{token}: {idf:.3f}")
```

### 10.4 Clustering Dokumentów

```python
# Obliczenie TF-IDF wektorów
vec = SimpleVectorizer(use_idf=True, norm='l2')
X = vec.fit_transform(documents)

# K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.predict(X)
```

---

## 11. Testy i Walidacja

Zaimplementowano 7 testów jednostkowych:

1. ✓ **Vocabulary Test** - Słownik zawiera unikalne tokeny
2. ✓ **Shape Test** - Macierz ma prawidłowy wymiar (n_docs, n_features)
3. ✓ **BoW Counts Test** - Liczby BoW są poprawnie zliczane
4. ✓ **Binary BoW Test** - Wartości binarne to 0 lub 1
5. ✓ **L2 Normalization Test** - Norma L2 każdego wektora ≈ 1.0
6. ✓ **min_df Filtering Test** - Rzadkie tokeny są filtrowane
7. ✓ **OOV Handling Test** - Nieznane tokeny są bezpiecznie obsługiwane

---

## 12. Przykłady Użycia

### Przykład 1: Prosty BoW

```python
corpus = [
    "Python jest wspaniałym językiem",
    "Machine learning to przyszłość",
    "Python i machine learning"
]

vec = SimpleVectorizer(binary=False, use_idf=False)
matrix = vec.fit_transform(corpus, return_df=True)
print(matrix)
```

### Przykład 2: TF-IDF z normalizacją

```python
vec = SimpleVectorizer(use_idf=True, norm='l2')
matrix = vec.fit_transform(corpus, return_df=True)
# Wszystkie wektory mają długość ~1.0
```

### Przykład 3: Obsługa Stoplisity

```python
vec = SimpleVectorizer(stop_words='english', use_idf=True)
vec.fit(corpus)
# Słowa takie jak "is", "the", etc. są usunięte
```

### Przykład 4: N-gramy

```python
vec = SimpleVectorizer(ngram_range=(1, 2))
vec.fit(corpus)
# Tokeny to unigramy ("Python") i bigramy ("machine learning")
```

---

## 13. Optymalizacje i Rozszerzenia

### Zastosowane Optymalizacje

1. **Vectorizacja NumPy**: Operacje na macierzach zamiast pętli
2. **Lazy IDF computation**: IDF obliczane tylko dla słownika
3. **Efektywne liczenie**: Counter z collections
4. **Copy-on-write**: Macierze skopiowane tylko gdy konieczne

### Możliwe Rozszerzenia

1. **Sparse matrices** (scipy.sparse.csr_matrix)
   - Oszczędność pamięci dla dużych zbiorów
   
2. **Incremental fit** (online learning)
   - Aktualizacja słownika bez ponownego trenowania
   
3. **Parallelizacja** (multiprocessing)
   - Przyspieszenie transformacji na dużych zbiorach
   
4. **Unicode Normalization** (NFD/NFC)
   - Lepsza obsługa znaków specjalnych

5. **Custom tokenizers**
   - Obsługa konkretnych przypadków (hashtagi, emojis)

---

## 14. Wnioski

Implementacja `SimpleVectorizer` demonstruje:

✓ Głębokie zrozumienie vectorizacji tekstu i TF-IDF  
✓ Prawidłowe obsługowanie edge cases (OOV, puste dokumenty)  
✓ Kompatybilność z interfejsem scikit-learn  
✓ Czysty, czytelny kod z dokumentacją  
✓ Kompleksowe testy walidacyjne  

Klasa jest:
- **Modułowa**: Łatwa do rozszerzenia
- **Efektywna**: Dobrze zoptymalizowana
- **Bezpieczna**: Obsługuje wiele wariantów użycia
- **Niezawodna**: Pokryta testami

---

## 15. Pliki Projektu

- `vectorizer.py` - Implementacja modułu SimpleVectorizer
- `Vectorizer_BoW_TF-IDF.ipynb` - Notebook z przykładami i testami
- `SPRAWOZDANIE.md` - Dokumentacja projektu

---

**Autor**: Implementacja własna  
**Data**: 2025  
**Język**: Python 3.10+  
**Zależności**: NumPy, Pandas, SciPy (opcjonalnie)