import re
import string

_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with",
}

def remove_whitespace(text):
    return text.strip()

def lowering(text):
    return text.lower()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])

def remove_stopword(text):
    tokens = re.findall(r"\b\w+\b", text)
    english_stopwords = _FALLBACK_STOPWORDS
    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]
    return " ".join(tokens_wo_stopwords)

def stem_words(text):
    # Not used in the current attack flow; keep a safe no-op placeholder.
    word_tokens = re.findall(r"\b\w+\b", text)
    return " ".join(word_tokens)
 
def text_preprocessing(text):
    text_1 = remove_whitespace(text)
    text_2 = lowering(text_1)
    text_3 = remove_numbers(text_2)
    text_4 = remove_punctuation(text_3)
    text_5 = remove_stopword(text_4)
    return text_5
