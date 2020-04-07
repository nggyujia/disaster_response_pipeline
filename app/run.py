import json
import plotly
from plotly.graph_objs import Bar
import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # for graph 1
    category_counts = pd.DataFrame(df.iloc[:, 4:].sum(axis = 0)).sort_values(0, ascending = False).reset_index(level=0)
    
    # for graph 2
    nltk.download('stopwords')
    df['tokens'] = df['message'].apply(lambda x: word_tokenize(re.sub(r"[^a-zA-Z0-9]"," ", x.lower())))
    stopwords_english = stopwords.words('english')
    stopwords_english.append('would')
    df['clean_tokens'] = df['tokens'].apply(lambda x: [w for w in x if w not in stopwords_english])
    
    clean_tokens = list()

    for row in df['clean_tokens']:
        for token in row:
            clean_tokens.append(token)
    
    word_counts = pd.DataFrame(pd.Series(clean_tokens).value_counts()).sort_values(0, ascending = False).reset_index(level = 0)

    # for graph 3
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_counts['index'],
                    y=category_counts[0],
                    marker_color='salmon'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },

        {
            'data': [
                Bar(
                    x=word_counts.iloc[:10]['index'],
                    y=word_counts.iloc[:10][0]
                )
            ],

            'layout': {
                'title': 'Top 10 Most Used Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color='orange'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()