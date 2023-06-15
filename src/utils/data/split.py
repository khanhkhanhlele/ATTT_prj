from sklearn.model_selection import train_test_split
from utils.tf_idf import TF_IDF

def BODMAS_split(df, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['category'], test_size=test_size)
    return x_train, x_test, y_train, y_test

def APISeq_split_TFIDF(df, test_size=0.12):
    words = list(map(lambda s: s.split(','), list(df['api'])))
    label = list(df['class'])
    tf_idf = TF_IDF(words)
    x_train, x_test, y_train, y_test = train_test_split(words, label, test_size=test_size)
    return x_train, x_test, y_train, y_test, tf_idf

def APISeq_split_token(df, test_size=0.1):
    words = list(map(lambda s: s.split(','), list(df['api'])))
    token = {}
    counter = 1
    for wordi in words:
        for wordii in wordi:
            if wordii not in token.keys():
                token[wordii] = counter
                counter += 1
    for i in range(len(words)):
        for ii in range(len(words[i])):
            words[i][ii] = token[words[i][ii]]
    label = list(df['class'])
    x_train, x_test, y_train, y_test = train_test_split(words, label, test_size=test_size)
    return x_train, x_test, y_train, y_test, counter

def APISeq_split_DB(df, test_size=0.1):
    words = list(map(lambda s: s.replace(',', ' '), list(df['api'])))
    label = list(df['class'])
    x_train, x_test, y_train, y_test = train_test_split(words, label, test_size=test_size)
    return x_train, x_test, y_train, y_test
