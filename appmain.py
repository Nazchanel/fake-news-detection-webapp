import sys

from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import contractions as cot
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import zipfile

app = Flask(__name__)  # Declare flask app


os.mkdir("archive8")

with zipfile.ZipFile("archive8.zip", "r") as zip_ref:
    zip_ref.extractall("./archive8")

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
nltk.download('punkt')

os.chdir("./FakeNews")  # Changes the directory to the folder with the csv files

fn = pd.read_csv("Fake.csv")
tn = pd.read_csv("True.csv")
fn['truth'] = 0  # Makes a column of 0s marking the data false
tn['truth'] = 1  # Makes a column of 1s marking the data true

tn.drop_duplicates(inplace=True)
fn.drop_duplicates(inplace=True)

# Import and processing/cleaning of the dataframe
extra = pd.read_csv("../archive8/politifact.csv")


# Drops the columns and rows that are not relevant
extra = extra.drop(
    columns=['Unnamed: 0', 'sources', 'sources_dates', 'sources_post_location', 'curator_name', 'curated_date',
             'curators_article_title', 'curator_complete_article', 'curator_tags', 'sources_url'])
extra.drop_duplicates(inplace=True)
extra.dropna(inplace=True)

# Replaces the truths we want with their corresponding binary value
extra['fact'].replace(['false', "pants-fire"], 0, inplace=True)
extra['fact'].replace(['true', 'mostly-true'], 1, inplace=True)

# Drops the rows of the truths we don't need
extra.drop(extra.loc[extra['fact'] == "half-true"].index, inplace=True)
extra.drop(extra.loc[extra['fact'] == "barely-true"].index, inplace=True)
extra.drop(extra.loc[extra['fact'] == "full-flop"].index, inplace=True)
extra.drop(extra.loc[extra['fact'] == "half-flip"].index, inplace=True)
extra.drop(extra.loc[extra['fact'] == "no-flip"].index, inplace=True)

extra.rename(columns={'sources_quote': 'title', 'fact': 'truth'}, inplace=True)


# Removes the \n's in the DataFrame
def remove_lines(text):
    text = text.strip("\n")
    return text


extra['title'] = extra['title'].apply(lambda x: remove_lines(x))

extra['text'] = extra['title']

# IMPORTANT: Balances the data; making the value higher will lean the program
# to predict true, lower is the opposite

fn = fn[:-2000]

fn.rename(columns={0: "title", 1: "text", 2: "subject", 3: "date", 4: "truth"}, inplace=True)

news = pd.concat([tn, fn, extra], axis=0, ignore_index=True)  # Combines the dataframes so its easier to work with

news.drop_duplicates(inplace=True)  # Drops any leftover duplicates
"""Preprocessing"""


def remove_contractions(text):
    fixed_word = []
    for word in text.split():
        fixed_word.append(cot.fix(word))
    counter = 0
    for i in fixed_word:
        if i != fixed_word[0]:
            counter += 1
        if i == "you.S.":
            fixed_word[counter] = "u.s."
        if i == "yous":
            fixed_word[counter] = "u.s."
    fixed_whole = ' '.join(fixed_word)
    return fixed_whole


# Applies the functions with lambda to do the stated function
news['title_wo_contra'] = news['title'].apply(lambda x: remove_contractions(x))
news['text_wo_contra'] = news['text'].apply(lambda x: remove_contractions(x))


def remove_punctuation(text):
    no_punct = [words for words in text if words not in string.punctuation]
    words_wo_punct = ''.join(no_punct)
    return words_wo_punct


# Applies the functions with lambda to do the stated function
news['title_wo_punct'] = news['title_wo_contra'].apply(lambda x: remove_punctuation(x))
news['text_wo_punct'] = news['text_wo_contra'].apply(lambda x: remove_punctuation(x))


def remove_stopwords(text):
    text = text.split()
    text = [word for word in text if word not in stopword]
    text = ' '.join(text)
    return text


# Applies the functions with lambda to do the stated function
news['title_wo_stopwords'] = news['title_wo_punct'].apply(lambda x: remove_stopwords(x.lower()))
news['text_wo_stopwords'] = news['text_wo_punct'].apply(lambda x: remove_stopwords(x.lower()))


# Removes any formatted quotation marks that the remove contractions function
# didn't remove

def remove_quotemarks(text):
    text = text.replace('“', "")
    text = text.replace('’', "")
    text = text.replace('”', "")
    return text


news['filtered_title'] = news['title_wo_stopwords'].apply(lambda x: remove_quotemarks(x))
news['filtered'] = news['text_wo_stopwords'].apply(lambda x: remove_quotemarks(x))

# Deletes all the excess columns and sets the title equal to the preprocessed version

news["joined_title"] = news["filtered_title"]
news = news.drop(["title_wo_contra", "title_wo_punct", "title_wo_stopwords", "filtered_title"], axis=1)
news["joined_text"] = news["filtered"]
news = news.drop(["text_wo_contra", "text_wo_punct", "text_wo_stopwords", "filtered"], axis=1)

"""# **Model**

## **Vectorization/Model**
"""

y = news['truth']
y = y.astype('int')  # Some of the y values are "objects", so this converts it to int
X = news['joined_text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Splits the data

# Pipeline makes it easy to predict; no direct vectorization needed
# Can be all applied in one line

text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(2, 3), binary=True)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(X_train, y_train)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/output', methods=['GET', 'POST'])
def foo():
    if request.method == 'POST':
        text_sample = request.form['test']
        text_sample = str(text_sample)
        text_sample = [text_sample]
        df = pd.DataFrame(text_sample, columns=['text'])

        def remove_contractions(text):
            fixed_word = []
            for word in text.split():
                fixed_word.append(cot.fix(word))
            counter = 0
            for i in fixed_word:
                if i != fixed_word[0]:
                    counter += 1
                if i == "you.S.":
                    fixed_word[counter] = "u.s."
                if i == "yous":
                    fixed_word[counter] = "u.s."
            fixed_whole = ' '.join(fixed_word)
            return fixed_whole

        df['text_wo_contra'] = df['text'].apply(lambda x: remove_contractions(x))

        def remove_punctuation(text):
            no_punct = [words for words in text if words not in string.punctuation]
            words_wo_punct = ''.join(no_punct)
            return words_wo_punct

        df['text_wo_punct'] = df['text_wo_contra'].apply(lambda x: remove_punctuation(x))

        def remove_stopwords(text):
            text = text.split()
            text = [word for word in text if word not in stopword]
            text = ' '.join(text)
            return text

        df['text_wo_punct_wo_stopwords'] = df['text_wo_punct'].apply(lambda x: remove_stopwords(x.lower()))

        def remove_quotemarks(text):
            text = text.replace('“', "")
            text = text.replace('’', "")
            text = text.replace('”', "")
            return text

        df['filtered'] = df['text_wo_punct_wo_stopwords'].apply(lambda x: remove_quotemarks(x))

        df["joined"] = df["filtered"]
        df = df.drop(["text_wo_contra", "text_wo_punct", "text_wo_punct_wo_stopwords", "filtered"], axis=1)
        tmp = df["joined"]
        text_sample = pd.Series.tolist(tmp)
        print(text_sample)
        sample_predict = text_clf.predict(text_sample)
        sample_predict = np.array(sample_predict)
        sample_predict = sample_predict.tolist()
        sample_predict = sample_predict[0]

        if text_sample[0] == '':
            for i in text_sample:
                sample_predict = i
        print(sample_predict)
        return sample_predict


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
