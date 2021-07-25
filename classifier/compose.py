import pandas as pd
from classifier.data_loader import loadData, upsampleData, getTestVectors
from flask import jsonify
import json

df = pd.read_csv("classifier/captions_p_df.csv", index_col=0)
df = df[df['caption'].apply(lambda x: isinstance(x, (str, bytes)))]
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

X = df['caption']
y = df['isEvent']

loaded_df, X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded, _, _, _, vectorizer, tfidfconverter = loadData()
X_train_r, X_test_r, y_train_1_r, y_test_1_r, vocab_size, tokenizer = upsampleData(loaded_df)

# import pickle
import joblib
# modelXGB = pickle.load(open("classifier/models/xgboost_spw_3.pkl", "rb"))
modelXGB = joblib.load("classifier/models/xgb.joblib.dat")

from tensorflow import keras
modelCNN = keras.models.load_model("classifier/models/cnn")
modelSimpleCNN = keras.models.load_model("classifier/models/cnn_simple")
modelBidir = keras.models.load_model("classifier/models/bidir")
import sys

try:
    modelSVC = pickle.load(open('classifier/models/svc', 'rb'))
except:
    print("Unexpected error:", sys.exc_info()[0])

import numpy as np
def predictLabels(X):
    try:
        X_test_vec, X_test_seqs = getTestVectors(X, vectorizer, tfidfconverter, tokenizer)
        predXGB = modelXGB.predict(X_test_vec)
        predCNN = np.argmax(modelCNN.predict(X_test_seqs), axis=1)
        predSimpleCNN = np.argmax(modelSimpleCNN.predict(X_test_seqs), axis=1)
        predBidir = np.argmax(modelBidir.predict(X_test_seqs), axis=1)
        predSVC = modelSVC.predict(X_test_vec)
        final_pred = np.zeros(predXGB.shape[0])
        for i in range(predXGB.shape[0]):
            value = (predXGB[i] + predCNN[i] + predSimpleCNN[i] + predBidir[i] + predSVC[i]) / 5
            value = round(value)
            final_pred[i] = value
    except:
        print("Unexpected error:", sys.exc_info()[0])

    # res = {'XGBoost': predXGB[0], 'CNN': predCNN[0], 'SimpleCNN': predSimpleCNN[0], 'BidirectionalGRU': predBidir[0], 'SVC': predSVC[0], 'avg': final_pred[0]}
    return json.dumps({'XGBoost': np.array2string(predXGB.astype('int'), separator=","), 'CNN': np.array2string(predCNN.astype('int'), separator=","), 'SimpleCNN': np.array2string(predSimpleCNN.astype('int'), separator=","), 'BidirectionalGRU': np.array2string(predBidir.astype('int'), separator=","), 'SVC': np.array2string(predSVC.astype('int'), separator=","), 'Average': np.array2string(final_pred.astype('int'), separator=",")})