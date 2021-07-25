import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

def loadData(useEmbed=True):
    df = pd.read_csv("classifier/captions_p_df.csv", index_col=0)
    df = df[df['caption'].apply(lambda x: isinstance(x, (str, bytes)))]
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    nltk.download('stopwords')
    nltk.download('wordnet')

    X = df['caption']
    y = df['isEvent']

    documents = []

    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('russian'))
    X = vectorizer.fit_transform(documents).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # train_x, valid_x, train_y, valid_y = train_test_split(df['caption'], df['isEvent'])

    embedding_matrix = []
    vocab_size = 0
    word_index = 0

    if(useEmbed):
        # create a tokenizer
        token = text.Tokenizer()
        token.fit_on_texts(df['caption'])
        word_index = token.word_index
        vocab_size = len(token.word_index) + 1  # Adding 1 because of reserved 0 index

    y_train_1 = y_train.astype('float32')
    y_test_1 = y_test.astype('float32')

    return df, X_train, X_test, y_train_1, y_test_1, embedding_matrix, vocab_size, word_index, vectorizer, tfidfconverter

def upsampleData(df):
    # Separate majority and minority classes
    data_majority = df[df['isEvent'] == 0]
    data_minority = df[df['isEvent'] == 1]

    bias = data_minority.shape[0] / data_majority.shape[0]
    # lets split train/test data first then
    train = pd.concat([data_majority.sample(frac=0.8, random_state=200),
                       data_minority.sample(frac=0.8, random_state=200)], ignore_index=True)
    test = pd.concat([data_majority.drop(data_majority.sample(frac=0.8, random_state=200).index),
                      data_minority.drop(data_minority.sample(frac=0.8, random_state=200).index)], ignore_index=True)

    train = train.sample(frac=1, random_state=200).reset_index(drop=True)
    test = test.sample(frac=1, random_state=200).reset_index(drop=True)


    # Separate majority and minority classes in training data for upsampling
    data_majority = train[train['isEvent'] == 0]
    data_minority = train[train['isEvent'] == 1]

    # Upsample minority class
    data_minority_upsampled = resample(data_minority,
                                       replace=True,  # sample with replacement
                                       n_samples=data_majority.shape[0],  # to match majority class
                                       random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    # sentences_train, sentences_test, y_train, y_test = train_test_split(df['caption'], df['isEvent'], test_size=0.2, random_state=0)
    sentences_train = data_upsampled['caption']
    sentences_test = test['caption']
    sentences_train = sentences_train.fillna('')
    sentences_test = sentences_test.fillna('')

    tokenizer = Tokenizer(num_words=5000)
    # tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_train)

    X_train_r = tokenizer.texts_to_sequences(sentences_train)
    X_test_r = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    maxlen = 100

    X_train_r = pad_sequences(X_train_r, padding='post', maxlen=maxlen)
    X_test_r = pad_sequences(X_test_r, padding='post', maxlen=maxlen)

    y_train_1 = data_upsampled['isEvent'].astype('float32')
    y_test_1 = test['isEvent'].astype('float32')

    return X_train_r, X_test_r, y_train_1, y_test_1, vocab_size, tokenizer

def getTestVectors(items, vectorizer, tfidfconverter, tokenizer):
    documents = []

    stemmer = WordNetLemmatizer()

    for sen in range(0, len(items)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(items[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    res_items = vectorizer.transform(documents).toarray()
    res_items = tfidfconverter.transform(res_items).toarray()

    maxlen = 100

    items_seqs = tokenizer.texts_to_sequences(items)
    items_seqs = pad_sequences(items_seqs, padding='post', maxlen=maxlen)

    return res_items, items_seqs