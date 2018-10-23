import os
import string
from functools import reduce

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class DataLoader(BaseEstimator, TransformerMixin):
    _contractions = {
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I shall / I will",
        "I'll've": "I shall have / I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
    }

    _stop_w = stopwords.words('english')
    _stop_w.remove('not')
    _the_trash = frozenset(
        _stop_w +
        list(string.punctuation) +
        ['\'s', 'br', 'film', 'movie', 'see', 'watch', 'one', 'two', 'three', 'four', 'five', 'six', 'try'] +
        list(string.digits)
    )

    @staticmethod
    def _read_file(path):
        with open(path, 'r', encoding="utf8") as f:
            raw_document = ' '.join([line.strip() for line in f])
        return raw_document

    def _filter(self, word):
        return any(not letter.isalpha() for letter in word) or word in self._the_trash

    @staticmethod
    def _expand_contractions(text, dic):
        """
        This method runs through text and replaces all contracted phrases with their full forms.
        For example: don't --> do not, shouldn't've --> should not have etc.
        :param text: original text with contractions.
        :param dic: a dictionary of type:
                            {
                                contraction1: full form1,
                                contraction2: full form2,
                                ...
                                contraction_N: full form_N
                            }
        :return: the text with full phrase forms.
        """

        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    @staticmethod
    def _lemmatize(tokens, lemmatizer, pos):
        return list(map(lambda x: lemmatizer.lemmatize(x, pos=pos), tokens))

    def _transf(self, path):
        from nltk.stem import WordNetLemmatizer

        word_net_lemmatizer = WordNetLemmatizer()

        raw_text = self._read_file(path)

        raw_text_without_contractions = self._expand_contractions(raw_text, self._contractions)

        tokenized_raw_text_without_contractions = word_tokenize(raw_text_without_contractions.lower())

        tokenized_raw_text_without_contractions_and_stop_words = [word for word in
                                                                  tokenized_raw_text_without_contractions if
                                                                  not self._filter(word)]

        lemmatized_tokenized_document = self._lemmatize(tokenized_raw_text_without_contractions_and_stop_words,
                                                        word_net_lemmatizer, 'v')

        lemmatized_tokenized_document = self._lemmatize(lemmatized_tokenized_document, word_net_lemmatizer, 'n')

        shrinked_lemmatized_tokenized_document = [token for token in lemmatized_tokenized_document if
                                                  len(token) > 2 and not self._filter(token)]

        return ' '.join(shrinked_lemmatized_tokenized_document)

    def fit(self, x, y=None):
        return self

    def transform(self, paths):
        return list(map(self._transf, paths))


def sample_file_addresses(pos_files_count, neg_files_count):
    pattern = './datasets/{}/{}/'
    pos_file_folders = [
        ['train', 'pos'],
        ['test', 'pos']
    ]
    neg_file_folders = [
        ['train', 'neg'],
        ['test', 'neg']
    ]
    pos_files = reduce(
        lambda x, z: x + z,
        [
            list(
                map(
                    lambda x: os.path.join(pattern.format(*folders), x),
                    os.listdir(pattern.format(*folders))
                )
            ) for folders in pos_file_folders]
    )
    neg_files = reduce(
        lambda x, z: x + z,
        [
            list(
                map(
                    lambda x: os.path.join(pattern.format(*folders), x),
                    os.listdir(pattern.format(*folders))
                )
            ) for folders in neg_file_folders]
    )
    pos_sample_ids = np.random.randint(len(pos_files), size=pos_files_count)
    neg_sample_ids = np.random.randint(len(neg_files), size=neg_files_count)
    pos_sample_files = np.array(pos_files)[pos_sample_ids]
    neg_sample_files = np.array(neg_files)[neg_sample_ids]
    X = np.array(pos_sample_files.tolist() + neg_sample_files.tolist())
    np.random.shuffle(X)
    y = np.array(list(map(lambda x: int('/pos/' in x), X)))
    return X, y
