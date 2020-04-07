# Disaster Response Pipeline Project

### Introduction
This project is part of the Udacity Data Science Nanodegree program. It combines knowledge on ETL and NLP and incorporates pipelines. The data is obtained through [Figure Eight](https://www.figure-eight.com/). 

### Files
You will find the raw data in the *data* folder as well as the python script to process the data. The processed data is then stored in a sqlite database created in the process_data.py script as well.

You will find the train_classifier.py script in the *models* folder. The script extracts the data from the sqlite database and performs training using NLP models from sklean library. See the script for the modules used. The model is then exported as a pickle file.

You will find the files related to rendering and deploying the results to a web interface in the *app* folder. The run.py script calls the model from the pickle file to predict user input and generate the graphs. 

### Instructions
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Improvements
The model can certainly be improved. More parameters can be considered through grid search, or other more sophisticated models can be used. There are many possibilities here.
