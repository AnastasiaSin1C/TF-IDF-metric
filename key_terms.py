from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk import pos_tag
from lxml import etree
from collections import Counter
from nltk.corpus import stopwords
import string
import heapq
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer




def stage1():
    n = 5
    xml_path = "news.xml"

    root = etree.parse(xml_path).getroot()
    corpus = root[0]

    news_dict = {}
    for elem in corpus:  # получить дочерние элементы

        head = elem[0].text
        text = elem[1].text

        news_dict[head] = common_tokens(text, n)

    print_common_words(news_dict)


def stage2():
    n = 5
    xml_path = "news.xml"

    root = etree.parse(xml_path).getroot()
    corpus = root[0]

    news_dict = {}
    for elem in corpus:  # получить дочерние элементы

        head = elem[0].text
        text = elem[1].text

        news_dict[head] = dict(Counter(process_text(text)).most_common(n))

    print_common_words(news_dict)

def stage3():
    n = 5
    xml_path = "news.xml"

    root = etree.parse(xml_path).getroot()
    corpus = root[0]

    news_dict = {}

    for elem in corpus:  # получить дочерние элементы

        head = elem[0].text
        text = elem[1].text

        words = process_text(text)
        nouns = [word for word in words if pos_tag([word])[0][1] == 'NN']
        news_dict[head] = dict(Counter(nouns).most_common(n))

    print_common_words(news_dict)

def stage4():
    n = 5
    xml_path = "news.xml"

    root = etree.parse(xml_path).getroot()
    corpus = root[0]

    texts = []
    heads = []
    for elem in corpus:  # получить дочерние элементы

        head = elem[0].text
        text = elem[1].text

        words = process_text(text)

        heads = heads + [head]
        texts = texts + [nouns_from_text(words)]

    top_words = get_top_5_words(texts)

    news_dict = dict(zip(heads, top_words))

    print_common_words(news_dict)


def get_top_5_words(texts):

    vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True,
                                 analyzer='word', ngram_range=(1, 1),
                                 stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    terms = vectorizer.get_feature_names_out()

    words = []
    for doc in tfidf_matrix:
        word_rates = {terms[ind]:doc[ind] for ind in np.argsort(doc)}
        words_top_5 = dict(sorted(word_rates.items(), key=lambda x: (x[1], x[0]), reverse=True)[:5])
        words = words + [words_top_5]

    return words


def nouns_from_text(words):
    nouns = [word for word in words if pos_tag([word])[0][1] == 'NN']
    return ' '.join(nouns)


def process_text(text):
    lemmatizer = WordNetLemmatizer()
    words = list(map(lambda x: lemmatizer.lemmatize(x), word_tokenize(text.lower())))
    stop_words = stopwords.words('english') + list(string.punctuation)
    words = [word for word in words if word not in stop_words]
    return sorted(words, reverse=True)


def print_common_words(news_dict):
    for head, common_words in news_dict.items():
        print(head + ':')
        print(*common_words.keys())
        print()


def common_tokens(text, n):
    text = sorted(word_tokenize(text.lower()), reverse=True)
    frequency = dict(Counter(text).most_common(n))
    return frequency


if __name__ == '__main__':
    stage4()
