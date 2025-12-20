"""
SimpleVectorizer - Implementacja wektoryzatorów BoW i TF-IDF

Moduł zawierający klasy do konwersji dokumentów tekstowych na wektory numeryczne
w reprezentacjach Bag-of-Words (BoW) i TF-IDF bez użycia scikit-learn.

Klasy:
    SimpleTokenizer: Tokenizacja i preprocesing tekstu
    SimpleVectorizer: Główny wektoryzator z metodami fit/transform

Autor: Implementacja własna
Data: 2025
"""

import numpy as np
import pandas as pd
import re
from collections import Counter
from typing import List, Optional, Dict, Tuple, Union
import math
import pickle
import json
from pathlib import Path
import warnings

# Opcjonalne: dla stemmingu
try:
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Dla normalizacji wektorów
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.sparse as sp_sparse


class SimpleTokenizer:
    """
    Klasa do tokenizacji i preprocesingu tekstu.
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 use_stemming: bool = False,
                 remove_diacritics: bool = False):
        """
        Args:
            lowercase: Konwersja tekstu na małe litery
            use_stemming: Zastosowanie stemowania (Porter Stemmer)
            remove_diacritics: Usuwanie znaków diakrytycznych
        """
        self.lowercase = lowercase
        self.use_stemming = use_stemming
        self.remove_diacritics = remove_diacritics
        
        if use_stemming and NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None
    
    def remove_accents(self, text: str) -> str:
        """Usuwanie znaków diakrytycznych"""
        import unicodedata
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    def tokenize(self, text: str, ngram_range: Tuple[int, int] = (1, 1)) -> List[str]:
        """
        Tokenizacja tekstu na słowa.
        
        Args:
            text: Tekst do tokenizacji
            ngram_range: Zakres n-gramów (min_n, max_n)
            
        Returns:
            Lista tokenów
        """
        # Preprocesing
        if self.remove_diacritics:
            text = self.remove_accents(text)
        
        if self.lowercase:
            text = text.lower()
        
        # Tokenizacja regex'em: wyodrębnianie słów (a-z, 0-9, _)
        tokens = re.findall(r'\w+', text)
        
        # Stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # N-gramy
        if ngram_range[0] == 1 and ngram_range[1] == 1:
            return tokens
        
        ngrams = []
        min_n, max_n = ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams


class SimpleVectorizer:
    """
    Wektoryzator BoW i TF-IDF implementowany od podstaw.
    
    Parametry:
    -----------
    lowercase : bool, default=True
        Konwersja tekstu na małe litery
    
    stop_words : list or 'english', optional
        Lista słów stopowych do usunięcia, lub 'english' dla standardowej listy
    
    min_df : int or float, default=1
        Ignoruj tokeny pojawiające się w mniej niż min_df dokumentach.
        Jeśli float (0-1), interpretuj jako odsetek dokumentów.
    
    max_df : int or float, default=1.0
        Ignoruj tokeny pojawiające się w więcej niż max_df dokumentach.
        Jeśli float (0-1), interpretuj jako odsetek dokumentów.
    
    max_features : int, optional
        Maksymalna liczba features (tokenów) do użycia
    
    binary : bool, default=False
        Jeśli True, wektory BoW zawierają 0/1 (obecność), nie liczby
    
    use_idf : bool, default=True
        Włącz obliczanie IDF dla TF-IDF
    
    smooth_idf : bool, default=True
        Dodaj 1 do df podczas obliczania IDF aby uniknąć dzielenia przez zero
    
    sublinear_tf : bool, default=False
        Zastosuj sublinearne skalowanie TF (tf = 1 + log(tf))
    
    norm : {None, 'l1', 'l2'}, default=None
        Normalizacja wektorów (L1 lub L2)
    
    handle_oov : {'ignore', 'add_column', 'error'}, default='ignore'
        Jak obsługiwać tokeny Out-Of-Vocabulary podczas transform
    
    use_stemming : bool, default=False
        Włącz Porter Stemming
    
    ngram_range : tuple of (min_n, max_n), default=(1, 1)
        Zakres n-gramów
    
    remove_diacritics : bool, default=False
        Usuwaj znaki diakrytyczne z tekstu
    
    Atrybuty po fit():
    ------------------
    vocabulary_ : dict
        Słownik {token -> indeks}
    
    idf_ : array, shape (n_features,)
        Computed IDF values
    
    document_frequencies_ : dict
        Liczność dokumentów dla każdego tokenu
    """
    
    def __init__(self,
                 lowercase: bool = True,
                 stop_words: Optional[Union[List[str], str]] = None,
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0,
                 max_features: Optional[int] = None,
                 binary: bool = False,
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False,
                 norm: Optional[str] = None,
                 handle_oov: str = 'ignore',
                 use_stemming: bool = False,
                 ngram_range: Tuple[int, int] = (1, 1),
                 remove_diacritics: bool = False):
        
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.binary = binary
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.handle_oov = handle_oov
        self.use_stemming = use_stemming
        self.ngram_range = ngram_range
        self.remove_diacritics = remove_diacritics
        
        # Tokenizer
        self.tokenizer = SimpleTokenizer(
            lowercase=lowercase,
            use_stemming=use_stemming,
            remove_diacritics=remove_diacritics
        )
        
        # Stoplista
        if stop_words == 'english':
            self.stop_words_list = self._get_english_stopwords()
        elif stop_words:
            self.stop_words_list = set(stop_words)
        else:
            self.stop_words_list = set()
        
        # State po fit
        self.vocabulary_ = None
        self.idf_ = None
        self.document_frequencies_ = None
        self.n_docs_fit_ = None
        self.is_fitted = False
    
    @staticmethod
    def _get_english_stopwords() -> set:
        """Wbudowana angielska stoplista"""
        english_stops = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
            'the', 'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
            'what', 'when', 'where', 'who', 'which', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'just', 'should', 'now'
        }
        return english_stops
    
    def fit(self, X: List[str]):
        """
        Buduje słownik tokenów z korpusu.
        
        Args:
            X : list of str
                Listę dokumentów (łańcuchów znakowych)
        
        Returns:
            self
        """
        n_docs = len(X)
        self.n_docs_fit_ = n_docs
        
        # Liczenie częstości dokumentów dla każdego tokenu
        doc_frequency = Counter()
        all_tokens = Counter()
        
        for doc in X:
            tokens = self.tokenizer.tokenize(doc, self.ngram_range)
            
            # Filtrowanie stoplisty
            tokens = [t for t in tokens if t not in self.stop_words_list]
            
            # Unikalne tokeny w dokumencie
            unique_tokens = set(tokens)
            doc_frequency.update(unique_tokens)
            
            # Liczenie wszystkich tokenów
            all_tokens.update(tokens)
        
        self.document_frequencies_ = dict(doc_frequency)
        
        # Filtrowanie wg min_df i max_df
        vocab = {}
        idx = 0
        
        # Konwersja min_df i max_df jeśli to procenty
        if isinstance(self.min_df, float):
            min_df_count = int(self.min_df * n_docs)
        else:
            min_df_count = self.min_df
        
        if isinstance(self.max_df, float):
            max_df_count = int(self.max_df * n_docs)
        else:
            max_df_count = self.max_df
        
        # Sortowanie tokenów wg liczby dokumentów (opcjonalnie po liczbie wystąpień)
        tokens_sorted = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)
        
        for token, freq in tokens_sorted:
            doc_freq = doc_frequency.get(token, 0)
            
            # Sprawdzenie kryteriów df
            if doc_freq < min_df_count or doc_freq > max_df_count:
                continue
            
            if self.max_features and idx >= self.max_features:
                break
            
            vocab[token] = idx
            idx += 1
        
        self.vocabulary_ = vocab
        
        # Oblicz IDF jeśli trzeba
        if self.use_idf:
            self._compute_idf(n_docs)
        
        self.is_fitted = True
        return self
    
    def _compute_idf(self, n_docs: int):
        """
        Oblicza wartości IDF dla wszystkich tokenów w słowniku.
        
        Wzór: IDF_t = log(N / (1 + n_t)) + 1  (z smooth_idf=True)
               lub  IDF_t = log(N / n_t)      (z smooth_idf=False)
        
        Gdzie:
            N = liczba dokumentów
            n_t = liczba dokumentów zawierających token t
        """
        idf = np.zeros(len(self.vocabulary_))
        
        for token, idx in self.vocabulary_.items():
            df = self.document_frequencies_.get(token, 1)
            
            if self.smooth_idf:
                idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1
            else:
                idf[idx] = math.log(n_docs / df) + 1
        
        self.idf_ = idf
    
    def transform(self, X: List[str], return_df: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Konwertuje dokumenty na wektory BoW lub TF-IDF.
        
        Args:
            X : list of str
                Lista dokumentów do transformacji
            
            return_df : bool
                Zwróć Pandas DataFrame zamiast NumPy array
        
        Returns:
            matrix : ndarray or DataFrame, shape (n_docs, n_features)
                Wektory dokumentów. Wiersze = dokumenty, kolumny = tokeny
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")
        
        n_docs = len(X)
        n_features = len(self.vocabulary_)
        
        # Inicjalizuj macierz
        matrix = np.zeros((n_docs, n_features), dtype=np.float32)
        oov_counts = np.zeros(n_docs)  # Liczba OOV tokenów na dokument
        
        for doc_idx, doc in enumerate(X):
            tokens = self.tokenizer.tokenize(doc, self.ngram_range)
            
            # Filtrowanie stoplisty
            tokens = [t for t in tokens if t not in self.stop_words_list]
            
            # Liczenie tokenów
            token_counts = Counter(tokens)
            
            # Przetwarzanie każdego tokenu
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    token_idx = self.vocabulary_[token]
                    
                    if self.binary:
                        matrix[doc_idx, token_idx] = 1
                    else:
                        matrix[doc_idx, token_idx] = count
                else:
                    # Obsługa OOV
                    if self.handle_oov == 'ignore':
                        pass
                    elif self.handle_oov == 'add_column':
                        oov_counts[doc_idx] += count
                    elif self.handle_oov == 'error':
                        warnings.warn(f"Token '{token}' not in vocabulary")
        
        # TF-IDF transformacja jeśli trzeba
        if self.use_idf:
            matrix = self._apply_tfidf(matrix)
        
        # Sublinear TF
        if self.sublinear_tf:
            matrix = self._apply_sublinear_tf(matrix)
        
        # Normalizacja
        if self.norm == 'l2':
            matrix = self._normalize_l2(matrix)
        elif self.norm == 'l1':
            matrix = self._normalize_l1(matrix)
        
        # Dodaj kolumnę OOV jeśli trzeba
        if self.handle_oov == 'add_column' and np.any(oov_counts > 0):
            oov_col = oov_counts.reshape(-1, 1)
            matrix = np.hstack([matrix, oov_col])
        
        # Zwróć jako DataFrame lub array
        if return_df:
            cols = list(self.vocabulary_.keys())
            if self.handle_oov == 'add_column' and np.any(oov_counts > 0):
                cols.append('<OOV>')
            return pd.DataFrame(matrix, columns=cols)
        
        return matrix
    
    def _apply_tfidf(self, matrix: np.ndarray) -> np.ndarray:
        """Zastosuj IDF scaling"""
        return matrix * self.idf_
    
    def _apply_sublinear_tf(self, matrix: np.ndarray) -> np.ndarray:
        """Zastosuj sublinear scaling: tf = 1 + log(tf)"""
        matrix = matrix.copy()
        matrix[matrix > 0] = 1 + np.log(matrix[matrix > 0])
        return matrix
    
    def _normalize_l2(self, matrix: np.ndarray) -> np.ndarray:
        """Normalizacja L2 (normą Euklidesową)"""
        matrix = matrix.copy()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Uniknij dzielenia przez zero
        return matrix / norms
    
    def _normalize_l1(self, matrix: np.ndarray) -> np.ndarray:
        """Normalizacja L1 (suma absolutnych wartości)"""
        matrix = matrix.copy()
        sums = np.abs(matrix).sum(axis=1, keepdims=True)
        sums[sums == 0] = 1  # Uniknij dzielenia przez zero
        return matrix / sums
    
    def fit_transform(self, X: List[str], return_df: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit i transform w jednym kroku.
        
        Args:
            X : list of str
                Lista dokumentów
            
            return_df : bool
                Zwróć Pandas DataFrame
        
        Returns:
            matrix : ndarray or DataFrame
        """
        return self.fit(X).transform(X, return_df=return_df)
    
    def get_feature_names_out(self) -> np.ndarray:
        """Zwróć nazwy features (tokeny) w kolejności"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer not fitted")
        
        # Sortuj wg indeksu
        sorted_vocab = sorted(self.vocabulary_.items(), key=lambda x: x[1])
        return np.array([token for token, _ in sorted_vocab])
    
    def save(self, path: str):
        """Zapisz stan vectorizatora do pliku"""
        state = {
            'vocabulary': self.vocabulary_,
            'idf': self.idf_,
            'document_frequencies': self.document_frequencies_,
            'n_docs_fit': self.n_docs_fit_,
            'params': {
                'lowercase': self.lowercase,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'max_features': self.max_features,
                'binary': self.binary,
                'use_idf': self.use_idf,
                'smooth_idf': self.smooth_idf,
                'sublinear_tf': self.sublinear_tf,
                'norm': self.norm,
                'handle_oov': self.handle_oov,
                'use_stemming': self.use_stemming,
                'ngram_range': self.ngram_range,
                'remove_diacritics': self.remove_diacritics,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Załaduj stan vectorizatora z pliku"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.vocabulary_ = state['vocabulary']
        self.idf_ = state['idf']
        self.document_frequencies_ = state['document_frequencies']
        self.n_docs_fit_ = state['n_docs_fit']
        
        for key, val in state['params'].items():
            setattr(self, key, val)
        
        self.is_fitted = True
        return self


if __name__ == "__main__":
    # Przykład użycia
    print("SimpleVectorizer - Example Usage")
    print("=" * 60)
    
    corpus = [
        "Python programming is fun",
        "Machine learning with Python",
        "Data science and Python",
    ]
    
    # BoW
    vec_bow = SimpleVectorizer(binary=False, use_idf=False)
    bow_matrix = vec_bow.fit_transform(corpus, return_df=True)
    print("\nBag-of-Words:")
    print(bow_matrix)
    
    # TF-IDF
    vec_tfidf = SimpleVectorizer(use_idf=True, norm='l2')
    tfidf_matrix = vec_tfidf.fit_transform(corpus, return_df=True)
    print("\nTF-IDF (z normalizacją L2):")
    print(tfidf_matrix.round(3))
