# Premier League Match Predictor

## Overview
This project is a Python-based application that utilizes neural networks (keras, tensorflow, scikit-learn, pandas, numpy) to predict the outcomes of Premier League matches. It uses historical match data from the 1995-96 season to train a multiple neural network models, which are later combined. This model can be used to make predictions for upcoming matches with an **accuracy of 61%**.

## Objective
Predict the probabilities of outcome (Home win, Draw, or Away win) of Premier League matches.

## Data Scraping
ALl data was retrieved from:
Evan Gower. Premier League Matches 1993-2023. Retrieved 2023-09-24 from [https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022](https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022)

## Data Preprocessing
Initial data included the following features:
- Season
- Date
- Matchday
- Home team
- Away team
- Home goals
- Away goals
- Full time result
This dataset was processed in the following ways:
- Conversion of columns to appropriate data types
- Addition of "date_int" column representing a float value for the date
- Normalization of "date_int" column to deal with large values
- Categorized team names and assigned them integer codes using one-hot encoding
- Replaced teams' string values with encoded integers
- Initial dataset included a row representing match details
- Created a dataset for each team, containing all matches played by that team (home and away):
  - Date_int (normalized date in float)
  - Venue (home/away)
  - Opponent
  - Goals scored by team
  - Goals scored by opponent
  - Full time result (win, draw, lose)
- Processed datasets for each team to improve accuracy by increasing number of features:
  - One-hot encoded opponent column into a set of binary columns
  - Added rolling data including goals scored, goals conceded, and full time result of last 3 matches

## Splitting into Train, Validation, and Test datasets
- Training and validation data: all matches from 1995-96 season to 2021-22 season. Train-validation dataset split was 80-20
- After using validation data set for hyperparameter tuning, models were retrained to fit training + validation dataset
- Training data: all matches of 2022-23 season

## Model Training Details
- Seperate models were built for each team in the league
- Tuning for hyperparameters was done for each team's model
- Features taken into consideration: 
  - date
  - home goals
  - away goals
  - venue
  - full time result
  - rolling data (goals scored, goals conceded, result) for each team to inculcate "form" aspect
  - data for all matches since 1995-96 season
- Model details:
  - Seperate model was built for each team
  - Hyperparameter tuning was done for all teams' models based on validation data set. Hyperparameters included:
    - amount of data considered for training
    - number of hidden layers
    - number of nodes in hidden layers
    - dropout probability
    - batch size
    - learning rate for optimizer
  - Best parameters were extracted based on least validation loss.
  - Validation dataset was concatenated with training dataset and all models were refitted with best hyperparameters for continuous model improvement
  - Matches were predicted based on both the home team and the away team models
  - Predictions from the home team and the away team models were merged to give a combined prediction with accuracy of 61%
