from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import pandas as pd
import re


def get_data(exclude=None, include=None):
    pattern = r'\(([A-Z]{3})\)'

    # Fichier brut
    input_file_path = 'data/train.txt'
    output_file_path = 'data/clean.txt'

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # Read each line from the input file
        for line in input_file:
            modified_line = re.sub(pattern, r'\1\t', line)
            output_file.write(modified_line)
    df = pd.read_csv('data/clean.txt', sep='\t', header=None)
    df.columns = ['Lang', 'Text']
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

    for a in y.unique():
        labels[a] = label_encoder.transform([a])[0]
    return label_encoder, labels

def get_train_dev_test(X, y):
    X_train, X_mid, y_train, y_mid = train_test_split(X, y, test_size=0.2, random_state=42)

    X_dev, X_test, y_dev, y_test = train_test_split(X_mid, y_mid, test_size=0.5, random_state=42)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

