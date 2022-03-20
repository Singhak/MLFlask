from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # df = pd.read_csv('spam.csv', encoding="latin-1")
    # df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    # X = df['v2']
    # y = df['label']
    # cv = CountVectorizer()
    # X = cv.fit_transform(X)  # Fit the Data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=42)
    # # Naive Bayes Classifier
    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    # joblib.dump((cv, clf), 'NB_spam_model.pkl')

    if request.method == 'POST':
        NB_spam_model = open('NB_spam_model.pkl', 'rb')
        cv, clf = joblib.load(NB_spam_model)
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        # print(data)
        # a if condition else b
        if(message):
            output = 'Spam' if my_prediction[0] == 1 else 'Normal'
            return render_template('home.html', prediction_text='This is a {} message'.format(output))
        return render_template('home.html', prediction_text='Please enter valid text meaasage.')


# if __name__ == '__main__':
#     app.run(debug=True)
