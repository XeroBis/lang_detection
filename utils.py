from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import plotly.express as px
import pandas as pd
import numpy as np
import re


def get_data(file_path=None, output_file_path=None, columns=None, exclude=None, include=None):
    pattern = r'\(([A-Z]{3})\)'
    if file_path is None:
    # Fichier brut
        input_file_path = 'data/train.txt'
    else:
        input_file_path = file_path
    
    if output_file_path is None:
        output_file_path = 'data/clean.txt'
    else:
        output_file_path = output_file_path

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # Read each line from the input file
        for line in input_file:
            modified_line = re.sub(pattern, r'\1\t', line)
            output_file.write(modified_line)
    df = pd.read_csv('data/clean.txt', sep='\t', header=None)
    if columns is None:
        df.columns = ['Lang', 'Text']
    else :
        df.columns = columns
    df.to_csv('data/train.csv', index=False)
    if exclude is not None:
        df = df[~df['Lang'].isin(exclude)]
    elif include is not None:
        df = df[df['Lang'].isin(include)]
    return df

def draw_confusion_matrix(y_test, y_pred, labels):
    labels_inverse = {labels[lang]:lang for lang in labels}
    y_pred = [labels_inverse[i] for i in y_pred]
    y_test = [labels_inverse[i] for i in y_test]
    lang_list =  [ k for k, v in labels.items()]
    data = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=lang_list)
    fig = px.imshow(data,
                    labels=dict(x="Languages Réel", y="Languages Prédit", color="confusion"),
                    x=lang_list,
                    y=lang_list,
                    text_auto=True,
                    aspect="auto")
    fig.update_xaxes(side="top")
    fig.show()
    return

def display_results(y_true, y_pred, labels, draw=False, average_param="macro"):

    if draw:
        draw_confusion_matrix(y_true, y_pred, labels)

    print("Accuracy ", accuracy_score(y_true, y_pred))

    print("Precision -", average_param)
    print("precision score ", precision_score(y_true, y_pred, average=average_param, zero_division=1))

    print("Recall -", average_param)
    print("recall score ",   recall_score(y_true, y_pred, average=average_param, zero_division=1))

    print("F1 -", average_param)
    print("f1 score ", f1_score(y_true, y_pred, average=average_param, zero_division=1))
    return

def get_label_encoder(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    labels = {}

    for a in np.unique(y):
        labels[a] = label_encoder.transform([a])[0]
    return label_encoder, labels

def get_train_dev_test(X, y):
    X_train, X_mid, y_train, y_mid = train_test_split(X, y, test_size=0.2)

    X_dev, X_test, y_dev, y_test = train_test_split(X_mid, y_mid, test_size=0.5)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def train_one_model(vectorizer, model, labels, label_encoder, X_train, y_train, X_dev, y_dev, X_test, y_test, draw=False):

    X_train_vect = vectorizer.fit_transform(X_train)
    X_dev_vect = vectorizer.transform(X_dev)
    X_test_vect = vectorizer.transform(X_test)

    y_train_labels = label_encoder.transform(y_train)
    y_dev_labels = label_encoder.transform(y_dev)
    y_test_labels = label_encoder.transform(y_test)
    scaler = preprocessing.StandardScaler(with_mean=False).fit(X_train_vect)
    X_scaled_train = scaler.transform(X_train_vect)

    model.fit(X_scaled_train, y_train_labels)

    X_dev_scaled = scaler.transform(X_dev_vect)
    y_pred_dev = model.predict(X_dev_scaled)
    accuracy_dev = accuracy_score(y_dev_labels, y_pred_dev)
    print(f"Accuracy DEV {vectorizer} et {model}: {accuracy_dev:.3f}")
    display_results(y_dev_labels, y_pred_dev, labels, draw)

    X_test_scaled = scaler.transform(X_test_vect)
    y_pred_test = model.predict(X_test_scaled)

    # Calcul de l'accuracy sur les données de test
    accuracy_test = accuracy_score(y_test_labels, y_pred_test)
    print(f"Accuracy TEST {vectorizer} et {model}: {accuracy_test:.3f}")
    display_results(y_test_labels, y_pred_test, labels, draw)

