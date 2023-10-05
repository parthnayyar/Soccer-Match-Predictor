#!/usr/bin/env python
# coding: utf-8

# In[766]:


# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime

warnings.filterwarnings("ignore")


# In[767]:


# read file which has game data from season 1995-96 to 2022-23

cols = ["season", "matchday", "date", "home", "h_goals", "a_goals", "away", "ftr"]
df = pd.read_csv("data.csv", names = cols)
df.head()


# In[768]:


# convert date from string to datetime
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

start = df.at[0, "date"].toordinal()

df["date_int"] = [df.at[i, "date"].toordinal()-start for i in range(len(df))]
# normalize date_int
df["date_int"] = (df["date_int"] - df["date_int"].min()) / (df["date_int"].max() - df["date_int"].min())

df.tail(10)


# In[769]:


# convert ftr: h=0, d=1, a=2

results = ["H", "D", "A"]

ftr = []

for i, row in df.iterrows():
    ftr.append(results.index(row["ftr"]))
    
df["ftr"] = ftr

df.head()


# In[770]:


# assign each team a code

teams = pd.unique(df["home"]).tolist()

h = []
a = []

for i, row in df.iterrows():
    h.append(teams.index(row["home"]))
    a.append(teams.index(row["away"]))
    
df["h"] = h
df["a"] = a

test_df = df[df["season"] == 2023]
df = df[df["season"] != 2023]

df.head()


# In[771]:


teams_dfs = []

# venue = 0 if home, 1 if away, result = 0  if win, 1 if draw, 2 is loss
team_df_cols = ["date_int", "opponent", "venue", "goals", "opp_goals", "result"]

for i in range(len(teams)):
    
    team_df = pd.DataFrame([], columns = team_df_cols)   
    teams_dfs.append(team_df)

for index, row in df.iterrows():
    teams_dfs[row["h"]].loc[len(teams_dfs[row["h"]])] = [row["date_int"], row["a"], 0, row["h_goals"], row["a_goals"], row["ftr"]]
    teams_dfs[row["a"]].loc[len(teams_dfs[row["a"]])] = [row["date_int"], row["h"], 1, row["a_goals"], row["h_goals"], 2-row["ftr"]]
    
for team_df in teams_dfs:
    team_df["opponent"] = (team_df["opponent"]).astype(int)
    team_df["venue"] = (team_df["venue"]).astype(int)
    team_df["goals"] = (team_df["goals"]).astype(int)
    team_df["opp_goals"] = (team_df["opp_goals"]).astype(int)
    team_df["result"] = (team_df["result"]).astype(int)


# In[772]:


# for i in range(len(teams)):
#     print(teams[i])
#     print(teams_dfs[i])


# In[773]:


def rolling_stats(df, cols, n):
    new_col_names = [col_name+"_"+str(i+1) for col_name in cols for i in range(n) ]
    new_col_vals = []
    for col in cols:
        for i in range(n):
            new_col_vals.append([])
    
    for index, row in df.iterrows():
        if index >= n:
            for col_i in range(len(cols)):
                for i in range(1, n+1):
                    new_col_vals[col_i*n + i - 1].append(df.at[index-i, cols[col_i]])
        
    df = df.iloc[n:]
    
    for i in range(len(new_col_names)):
        df[new_col_names[i]] = new_col_vals[i]
    
    return df


# In[774]:


rolling_avg_cols = ["goals", "opp_goals", "result"]

teams_dfs_new = []

for team_df in teams_dfs:
    team_df_new = team_df.copy(deep=True)
    # add rolling stats
    team_df_new = rolling_stats(team_df_new, rolling_avg_cols, 3)
    # one hot encoding for opponent
    team_df_new = pd.get_dummies(team_df_new, columns = ["opponent"], dtype=int)
    # add to list
    teams_dfs_new.append(team_df_new)


# In[775]:


# for i in range(len(teams)):
#     print(teams[i])
#     print(teams_dfs_new[i])


# In[776]:


from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[777]:


train_dfs = []
valid_dfs = []

for team_df in teams_dfs_new:
    train, valid = train_test_split(team_df, test_size=0.2, shuffle=False)
    train_dfs.append(train)
    valid_dfs.append(valid)


# In[778]:


Xs = []

for team_df in teams_dfs_new:
    cols = list(team_df.columns)
    X = list(filter(lambda x: x not in ["goals", "opp_goals", "result"], cols))
    Xs.append(X)
    
# y = ["goals", "opp_goals", "result"]
y = "result"


# In[535]:


def plot_history(history):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(history.history["loss"], label="loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Categorical crossentropy")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(history.history["accuracy"], label="accuracy")
    ax2.plot(history.history["val_accuracy"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()
    
    plt.show()


# In[603]:


def train_model(X_train, y_train, X_valid, y_valid, num_layers, num_nodes, dropout_prob, learning_rate, batch_size, epochs):    
    nn_model = tf.keras.Sequential()
    
    nn_model.add(tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(len(X_train.columns),)))
    
    for i in range(num_layers):
        nn_model.add(tf.keras.layers.Dropout(dropout_prob))
        nn_model.add(tf.keras.layers.Dense(num_nodes, activation="relu"))
        
    nn_model.add(tf.keras.layers.Dense(3, activation="softmax"))
    
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    
    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False, validation_data=(X_valid, y_valid), use_multiprocessing=True)
    
    return nn_model, history


# In[779]:


i = 1

train_df_temp = train_dfs[i]
l = len(train_df_temp)
train_df = train_df_temp.iloc[int(l*0.75):]

X_train = train_df[Xs[i]]
y_train = pd.get_dummies(train_df[y])

valid_df = valid_dfs[i]
X_valid = valid_df[Xs[i]]
y_valid = pd.get_dummies(valid_df[y])

model, history = train_model(X_train, y_train, X_valid, y_valid, 1, 32, 0.3, 0.0001, 19, 100)
plot_history(history)

model.evaluate(X_valid, y_valid)


# In[780]:


teams_2023 = test_df["home"].unique()
teams_2023 = [teams.index(team) for team in teams_2023]

teams_2023


# In[608]:


jobs = 6 # it means number of cores
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=jobs,
                         inter_op_parallelism_threads=jobs,
                         allow_soft_placement=True,
                         device_count={'CPU': jobs})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[611]:


# train all teams models

models = {}
settings = {}

for i in teams_2023:
    
    least_val_loss = float("inf")
    least_val_model = None
    least_val_settings = {}
    epochs = 100
    
    train_df_temp = train_dfs[i]
    l = len(train_df_temp)
    
    valid_df = valid_dfs[i]
    X_valid = valid_df[Xs[i]]
    y_valid = pd.get_dummies(valid_df[y])
    
    print(i, teams[i], end = ": ")
    
    for split in [0,0.25,0.5]:
        
        train_df = train_df_temp.iloc[int(l*split):]
        X_train = train_df[Xs[i]]
        y_train = pd.get_dummies(train_df[y])
        
        for layers in [1, 2]:
            for num_nodes in [32, 64]:
                for dropout_prob in [0.1, 0.2]:
                    for learning_rate in [0.001, 0.0001]:
                        for batch_size in [12, 19, 38]:
                            model, history = train_model(X_train, y_train, X_valid, y_valid, layers, num_nodes, dropout_prob, learning_rate, batch_size, epochs)
#                             plot_history(history)
                            val_loss = model.evaluate(X_valid, y_valid)[0]
                            if val_loss < least_val_loss:
                                least_val_loss = val_loss
                                least_val_model = model
                                least_val_settings["split"] = split
                                least_val_settings["layers"] = layers
                                least_val_settings["num_nodes"] = num_nodes
                                least_val_settings["dropout_prob"] = dropout_prob
                                least_val_settings["learning_rate"] = learning_rate
                                least_val_settings["batch_size"] = batch_size
                                
    models[i] = least_val_model
    settings[i] = least_val_settings
    print("done")


# In[1076]:


train_dfs_new = []

for i in range(len(train_dfs)):
    train_dfs_new.append(pd.concat([train_dfs[i], valid_dfs[i]], ignore_index=True))


# In[954]:


def train_model_without_validation(X_train, y_train, num_layers, num_nodes, dropout_prob, learning_rate, batch_size, epochs):    
    nn_model = tf.keras.Sequential()
    
    nn_model.add(tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(len(X_train.columns),)))
    
    for i in range(num_layers):
        nn_model.add(tf.keras.layers.Dropout(dropout_prob))
        nn_model.add(tf.keras.layers.Dense(num_nodes, activation="relu"))
        
    nn_model.add(tf.keras.layers.Dense(3, activation="softmax"))
    
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    
    nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False, use_multiprocessing=True)
    
    return nn_model


# In[957]:


models_new = {}

for i in models:
    print(teams[i])
    
    split = settings[i]["split"]
    layers = settings[i]["layers"]
    num_nodes = settings[i]["num_nodes"]
    dropout_prob = settings[i]["dropout_prob"]
    learning_rate = settings[i]["learning_rate"]
    batch_size = settings[i]["batch_size"]
    
    epochs = 100
    
    train_df_temp = train_dfs_new[i]
    l = len(train_df_temp)
    
    train_df = train_df_temp.iloc[int(l*split):]
    X_train = train_df[Xs[i]]
    y_train = pd.get_dummies(train_df[y])
        
    models_new[i] = train_model_without_validation(X_train, y_train, layers, num_nodes, dropout_prob, learning_rate, batch_size, epochs)


# In[781]:


teams_dfs_test = {}

team_df_cols = ["date", "date_int", "opponent", "venue", "goals", "opp_goals", "result"]

for i in teams_2023:
    
    team_df_test = pd.DataFrame([], columns = team_df_cols)
    team_df_test["date"] = pd.to_datetime(team_df_test["date"])
    team_df_test["date_int"] = pd.to_numeric(team_df_test["date_int"])
    team_df_test["opponent"] = (team_df_test["opponent"]).astype(int)
    team_df_test["venue"] = (team_df_test["venue"]).astype(int)
    team_df_test["goals"] = (team_df_test["goals"]).astype(int)
    team_df_test["opp_goals"] = (team_df_test["opp_goals"]).astype(int)
    team_df_test["result"] = (team_df_test["result"]).astype(int)
    
    teams_dfs_test[i] = team_df_test

for index, row in test_df.iterrows():
    teams_dfs_test[row["h"]].loc[len(teams_dfs_test[row["h"]])] = [row["date"], row["date_int"], row["a"], 0, row["h_goals"], row["a_goals"], row["ftr"]]
    teams_dfs_test[row["a"]].loc[len(teams_dfs_test[row["a"]])] = [row["date"], row["date_int"], row["h"], 1, row["a_goals"], row["h_goals"], 2-row["ftr"]]


# In[783]:


teams_dfs_test_new = {}

for i in teams_dfs_test:
    team_df_test_new =  teams_dfs_test[i].copy(deep=True)
    # add rolling stats
    team_df_test_new = rolling_stats(team_df_test_new, rolling_avg_cols, 3)
    # one hot encoding for opponent
    team_df_test_new = pd.get_dummies(team_df_test_new, columns = ["opponent"], dtype=int)
    # add to list
    teams_dfs_test_new[i] = team_df_test_new


# In[796]:


for i in teams_dfs_test_new:
    for col in teams_dfs_new[i].columns:
        if col not in teams_dfs_test_new[i].columns:
            teams_dfs_test_new[i][col] = (np.zeros(len(teams_dfs_test_new[i]), )).astype(int)


# In[1055]:


def make_prediction(home, away, model=0):
    
    home_i = teams.index(home)
    away_i = teams.index(away)
    
    home_df = teams_dfs_test_new[home_i]
    away_df = teams_dfs_test_new[away_i]
    
    home_row = home_df[(home_df["opponent_"+str(away_i)] == 1) & (home_df["venue"] == 0)]
    away_row = away_df[(away_df["opponent_"+str(home_i)] == 1) & (away_df["venue"] == 1)]
    
    home_X = home_row[Xs[home_i]]
    away_X = away_row[Xs[away_i]]
    
    home_model = models_new[home_i]
    away_model = models_new[away_i]
    
    home_pred = home_model.predict(home_X, verbose=0)
    away_pred = away_model.predict(away_X, verbose=0)
    
    if model == 0:
        home_win = (home_pred[0][0] + away_pred[0][2])*100/2
        draw = (home_pred[0][1] + away_pred[0][1])*100/2
        away_win = (home_pred[0][2] + away_pred[0][0])*100/2    

        pred_prob = [home_win, draw, away_win]
        pred_max_i = pred_prob.index(max(pred_prob))

        return pred_max_i
    
    else:
        if home_pred[0][0] - away_pred[0][0] > 0.06:
            return 0
        elif home_pred[0][0] - away_pred[0][0] < -0.04:
            return 2
        else:
            return 1


# In[1081]:


test_df_new = test_df[test_df["matchday"] > 3]


# In[1082]:


predictions = []

for index, row in test_df_new.iterrows():
    predictions.append(make_prediction(row["home"], row["away"]))


# In[979]:


from sklearn.metrics import classification_report


# In[1025]:


actual = list(test_df_new["ftr"].values)
target_names = ["Home win or Draw", "Away win"]


# In[1084]:


print(classification_report(actual, predictions, target_names=["Home win", "Draw", "Away win"]))
print("             h  d   a")
print("Actual", end=":     ")
print(actual.count(0), end=" ")
print(actual.count(1), end=" ")
print(actual.count(2))
print("Prediction", end=": ")
print(predictions.count(0), end=" ")
print(predictions.count(1), end=" ")
print(predictions.count(2))


# In[1066]:


predictions_new = []
actual_new = []

for i in range(len(actual)):
    if actual[i] == 0 or actual[i] == 1:
        actual_new.append(0)
    else:
        actual_new.append(1)
        
    if predictions[i] == 0 or predictions[i] == 1:
        predictions_new.append(0)
    else:
        predictions_new.append(1)


# In[1072]:


print(classification_report(actual_new, predictions_new, target_names=target_names))
print("            h/d  a")
print("Actual", end=":     ")
print(actual_new.count(0), end=" ")
print(actual_new.count(1))
print("Prediction", end=": ")
print(predictions_new.count(0), end=" ")
print(predictions_new.count(1))
