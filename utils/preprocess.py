import re
import logging
from typing import List, Tuple, Optional
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from .stopwords_lv import STOPWORDS

# Try to import LatvianNLP, provide fallback if not available
try:
    from LatvianNLP import MorphoTagger
    morpho = MorphoTagger()
    LATVIAN_NLP_AVAILABLE = True
except ImportError:
    logging.warning("LatvianNLP not available. Lemmatization will use fallback method.")
    morpho = None
    LATVIAN_NLP_AVAILABLE = False

# Initialize stemmer
stemmer = SnowballStemmer('latvian', ignore_stopwords=False)

# 1. Lielo/mazo burtu normalizācija (lowercasing)
def normalize_case(text: str, casefold: bool = False) -> str:
    """
    Convert text to lowercase (or full Unicode casefold).
    - casefold=False: use str.lower() (default as per thesis)
    - casefold=True: use str.casefold() for aggressive Unicode folding
    """
    return text.casefold() if casefold else text.lower()

# 2. Pieturzīmju un skaitļu tīrīšana
def clean_punct_and_numbers(text: str, keep_numbers: bool = True, keep_exclamation: bool = True) -> str:
    """
    Remove URLs and unwanted symbols; keep Latvian letters and optionally digits/exclamation marks.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Build pattern based on what to keep
    pattern = r'[^āčēģīķļņōŗšūža-z\s'
    if keep_numbers:
        pattern = pattern[:-1] + r'0-9'
    if keep_exclamation:
        pattern = pattern[:-1] + r'!'
    pattern += ']'
    
    # Replace unwanted characters with space
    return re.sub(pattern, ' ', text)

# 3. Stopvārdu noņemšana
def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filter out tokens in the custom Latvian STOPWORDS list.
    Note: 'kas' and 'kā' are not in stopwords as they're clickbait indicators.
    """
    return [t for t in tokens if t not in STOPWORDS]

# Helper: tokenization (used in steps 3 & 4)
def tokenize(text: str) -> List[str]:
    """
    Split text into word tokens using NLTK's tokenizer.
    Note: Ensure you have downloaded NLTK's punkt tokenizer:
    >>> import nltk
    >>> nltk.download('punkt')
    """
    try:
        return word_tokenize(text, language='latvian')
    except LookupError:
        # Fallback to simple tokenization if Latvian tokenizer not available
        logging.warning("Latvian tokenizer not available. Using simple split.")
        return text.split()

# 4a. Lemmatizēšana
def lemmatize(tokens: List[str]) -> List[str]:
    """
    Morphological lemmatization via MorphoTagger (preferred over stemming).
    """
    if LATVIAN_NLP_AVAILABLE and morpho:
        return [morpho.get_lemma(t) for t in tokens]
    else:
        # Fallback to stemming if LatvianNLP not available
        logging.warning("Using stemming as fallback for lemmatization")
        return stem_tokens(tokens)

# 4b. Stemming
def stem_tokens(tokens: List[str]) -> List[str]:
    """
    SnowballStemmer for Latvian (alternative to lemmatization).
    """
    return [stemmer.stem(t) for t in tokens]

# 4c. Bez morfoloģiskās normalizācijas
def identity(tokens: List[str]) -> List[str]:
    """
    Return tokens unchanged (neither lemmatize nor stem).
    """
    return tokens

# 5-8. TF-IDF vectorizer factory
def get_vectorizer(
    max_features: Optional[int] = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Create a TfidfVectorizer with parameters optimized according to thesis:
      - max_features: 1000 (best F1 results)
      - ngram_range: (1,2) for unigrams and bigrams
      - min_df: 2 (removes words appearing only once)
      - sublinear_tf: True (logarithmic scaling improved F1 by 0.6%)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        token_pattern=r'\b\w+\b'  # Ensure proper tokenization
    )

# Utility: vectorize preprocessed texts
def vectorize_texts(
    texts: List[str],
    vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple:
    """
    Fit-transform a list of preprocessed strings into TF-IDF features.
    Returns: (X_matrix, fitted_vectorizer)
    """
    if vectorizer is None:
        vectorizer = get_vectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Full preprocessing pipeline (steps 1-4 + join)
def preprocess_text(
    text: str,
    casefold: bool = False,
    morpho_method: str = 'lemmatize',
    keep_numbers: bool = True,
    keep_exclamation: bool = True
) -> str:
    """
    Apply Steps 1-4 to a single text string, then join into one cleaned string.
    According to thesis, best configuration:
    - casefold=False (use lower())
    - morpho_method='lemmatize'
    - keep_numbers=True
    - keep_exclamation=True
    
    morpho_method options:
      - 'lemmatize' (default, best results)
      - 'stem'
      - 'identity' (no morphological normalization)
    """
    # Step 1: Case normalization
    t = normalize_case(text, casefold=casefold)
    
    # Step 2: Clean punctuation and numbers
    t = clean_punct_and_numbers(t, keep_numbers=keep_numbers, keep_exclamation=keep_exclamation)
    
    # Step 3: Tokenize and remove stopwords
    toks = tokenize(t)
    toks = remove_stopwords(toks)
    
    # Step 4: Morphological normalization
    if morpho_method == 'lemmatize':
        toks = lemmatize(toks)
    elif morpho_method == 'stem':
        toks = stem_tokens(toks)
    elif morpho_method == 'identity':
        toks = identity(toks)
    else:
        raise ValueError(f"Unknown morpho_method: {morpho_method}")
    
    return ' '.join(toks)