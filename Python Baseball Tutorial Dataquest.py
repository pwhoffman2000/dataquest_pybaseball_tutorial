#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import pandas as pd
from pybaseball import batting_stats
import numpy as np


# In[15]:


START = 2002
END = 2022


# In[16]:


batting = batting_stats(START, END, qual=200)


# In[17]:


batting.to_csv("batting.csv")


# In[18]:


# Getting rid of players with only 1 szn, as we need at least 2 szns to predict
batting = batting.groupby("IDfg", group_keys = False).filter(lambda x: x.shape[0] > 1)


# In[19]:


batting


# In[21]:


# We are trying to predict target (WAR) for the following season 
def next_season(player):
    player = player.sort_values("Season")
    player["Next_WAR"] = player["WAR"].shift(-1)
    return player

batting = batting.groupby("IDfg", group_keys=False).apply(next_season)


# In[22]:


batting[["Name", "Season", "WAR", "Next_WAR"]]


# In[24]:


#Cleaning data, getting rid of null values

#Counting missing values in each column
null_count = batting.isnull().sum()

null_count
    


# In[25]:


# Creating list where all values are complete
complete_cols = list(batting.columns[null_count == 0])


# In[26]:


complete_cols


# In[27]:


batting = batting[complete_cols + ["Next_WAR"]].copy()


# In[28]:


batting


# In[29]:


batting.dtypes


# In[30]:


#We need to handle strings as they will not work with ML algorithms
batting.dtypes[batting.dtypes == "object"]


# In[31]:


batting["Dol"]


# In[32]:


del batting["Dol"]


# In[33]:


batting["Age Rng"]


# In[34]:


del batting["Age Rng"]


# In[36]:


#Coding team names so we can use in algorithm
batting["team_code"] = batting["Team"].astype("category").cat.codes


# In[40]:


batting_full = batting.copy()
batting = batting.dropna()


# In[38]:


#Selecting useful features, don't want to overfit
#Running feature selector

from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

#higher alpha or lambda reduces overfitting, lower is close to lin reg
rr = Ridge(alpha=1)

split = TimeSeriesSplit(n_splits=3)
#Cross Validation is cv, going through all the features and find the best one
sfs = SequentialFeatureSelector(rr, n_features_to_select=20, direction = "forward", cv=split, n_jobs=4)


# In[39]:


#Before we run feature selector we need to make sure certain columns aren't passed
removed_columns = ["Next_WAR", "Name", "Team", "IDfg", "Season"]
selected_columns = batting.columns[~batting.columns.isin(removed_columns)]


# In[41]:


#Scaling data so the mean is 0 and stdev is 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
batting.loc[:, selected_columns] = scaler.fit_transform(batting[selected_columns])


# In[42]:


batting


# In[47]:


#MinMax scaler scales all values to be between 0 and 1
batting.describe()


# In[48]:


sfs.fit(batting[selected_columns], batting["Next_WAR"])


# In[46]:


sfs.get_support()


# In[51]:


#We want to index our selected columns list
predictors = list(selected_columns[sfs.get_support()])


# In[54]:


#Need to backtest, generate predictions
def backtest(data, model, predictors, start = 5, step=1):
    all_predictions = []
    
    years = sorted(batting["Season"].unique())
    
    for i in range(start, len(years), step):
        current_year = years[i]
        
        train = data[data["Season"] < current_year]
        test = data[data["Season"] == current_year]
        
        model.fit(train[predictors], train["Next_WAR"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["Next_WAR"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)


# In[55]:


predictions = backtest(batting, rr, predictors)


# In[57]:


predictions


# In[58]:


from sklearn.metrics import mean_squared_error

mean_squared_error(predictions["actual"], predictions["prediction"])


# In[59]:


batting["Next_WAR"].describe()


# In[60]:


2.8008640414488855 ** 0.5


# In[62]:


#Improving Accuracy, function called Player History

def player_history(df):
    df = df.sort_values("Season")
    
    df["player_season"] = range(0, df.shape[0])
    df["war_corr"] = list(df[["player_season", "WAR"]].expanding().corr().loc[(slice(None), "player_season"), "WAR"])
    df["war_corr"].fillna(1, inplace=True)
    
    df["war_diff"] = df["WAR"] / df["WAR"].shift(1)
    df["war_diff"].fillna(1, inplace = True)
    
    df["war_diff"][df["war_diff"] == np.inf] = 1
    
    return df

batting = batting.groupby("IDfg", group_keys=False).apply(player_history)


# In[63]:


def group_averages(df):
    return df["WAR"] / df["WAR"].mean()


# In[64]:


batting["war_season"] = batting.groupby("Season", group_keys=False).apply(group_averages)


# In[65]:


new_predictors = predictors + ["player_season", "war_corr", "war_season", "war_diff"]


# In[66]:


predictions = backtest(batting, rr, predictors)


# In[67]:


mean_squared_error(predictions["actual"], predictions["prediction"])


# In[72]:


pd.Series(rr.coef_, index = predictors).sort_values()


# In[73]:


diff = predictions["actual"] - predictions["prediction"]


# In[74]:


merged = predictions.merge(batting, left_index = True, right_index=True)


# In[75]:


merged["diff"] = (predictions["actual"] - predictions["prediction"]).abs()


# In[76]:


merged


# In[78]:


merged[["IDfg", "Season", "Name", "WAR", "Next_WAR", "diff"]].sort_values(["diff"])


# In[ ]:




