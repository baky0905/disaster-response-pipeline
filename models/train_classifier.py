#import string
import multiprocessing
from functools import partial
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists
import pickle


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import parallel_backend

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

stop_words = set(stopwords.words('english'))


def load_data(database_filepath):
    '''
    Input:
        database_filename(str): Filepath of the database.
    Output:
        X(numpy.ndarray): Array of input features.
        y(numpy.ndarray): Output labels, classes.
    '''

    try:
        database_exists(f'sqlite:///{database_filepath}')
        engine = create_engine(f'sqlite:///{database_filepath}')
        connection = engine.connect()

        df = pd.read_sql_table("messages_categories", con=connection)
        labels = df.iloc[:, 4:].columns

        X = df["message"].values
        y = df.iloc[:, 4:].values

        connection.close()

        return X, y, labels

    except:
        print("Database does not exist! Check your database_filepath!")


def tokenize(text):
    ''' Normalize, lemmantize and tokenize text messages.
    Input:
        text(str): Text messages.
    Output:
        clean_tokens(str): Normalize, lemmantize and tokenize text messages.
    '''

    # normalize text
    normalized_text = text.lower().strip()

    # tokenize text
    tokens = word_tokenize(normalized_text)

    # lemmantize text and remove stop words and non alpha numericals
    clean_tokens = []
    for token in tokens:
        lemmatizer = WordNetLemmatizer()
        clean_token = lemmatizer.lemmatize(token)

        if clean_token not in stop_words and clean_token.isalpha():
            clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    '''Build a Machine Learning pipeline using TfidfTransformer, RandomForestClassifier and GridSearchCV
    Input: 
        None
    Output:
        cv(sklearn.model_selection._search.GridSearchCV): Results of GridSearchCV
    '''

    text_clf = Pipeline([
                        ('vect', CountVectorizer(tokenizer=partial(tokenize))),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(
                            estimator=RandomForestClassifier(verbose=1)))
                        ])

    # parameters = {
    #    'clf__estimator__max_depth': [4, 6, 10, 12],
    #    'clf__estimator__n_estimators': [20, 40, 100],
    # }

    # grid_fit = GridSearchCV(
    #    estimator=text_clf,
    #    param_grid=parameters,
    #    verbose=3,
    #    cv=2,
    #    n_jobs=-1)

    return text_clf


def evaluate_model(model, X_test, y_test, labels):
    """ Function that will predict on X_test messages using build_model() function that
    transforms messages, extract features and trains a classifer.

    Input:
        model(sklearn.model_selection._search.GridSearchCV): Trained model.
        X_test(numpy.ndarray): Numpy array of messages that based on which trained model will predict.
        y_test(numpy.ndarray): Numpy array of classes that will be used to validate model predictions.
        labels(pandas.core.indexes.base.Index): Target labels for a multiclass prediction.

    Output:
        df(pandas.core.frame.DataFrame): Dataframe that contains report showing the main classification metrics.
    """
    y_pred = model.predict(X_test)

    df = pd.DataFrame(classification_report(
        y_test, y_pred, target_names=labels, output_dict=True)).T.reset_index()
    df = df.rename(columns={"index": "labels"})

    return df


def save_model(model, filepath):
    '''Saves the model to defined filepath
    Input 
        model(sklearn.model_selection._search.GridSearchCV): The model to be saved.
        model_filepath(str): Filepath  where the model will be saved.
    Output
        This function will save the model as a pickle file on the defined filepath.
    '''
    temporary_pickle = open(filepath, 'wb')
    pickle.dump(model, temporary_pickle)
    temporary_pickle.close()
    print("Model has been succesfully saved!")


# def main():
# https://github.com/scikit-learn/scikit-learn/issues/10533

if __name__ == '__main__':
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, Y_train)

            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
