# Dance to the Music

By: Jason Tellez

Date: October 14, 2021

The goal of this report is to find the drivers of a song's danceability score according to Spotify. Specifically, we will be looking at Spotify's top worldwide songs of the later portion on 2019, using the features provided by Spotify, and predicting a song's danceability score by creating machine learning models. The data was gathered and posted on Kaggle by user [rafaelnduarte](https://www.kaggle.com/rafaelnduarte/spotify-data-with-audio-features), who created the csv by using [Spotify's Web API](https://developer.spotify.com/documentation/web-api/).


## Table of contents

- [Table of contents](#table-of-contents)
- [Project Summary](#project-summary)
- [Executive Summary](#executive-summary)
- [Dictionary](#dictionary)
- [Pipeline](#pipeline)
- [Conclusions](#conclusions)   
- [Recreate these results](#recreate-these-results)


## Project Summary
[(Back to top)](#table-of-contents)

#### Goal
The goal is to use linear regression algorithms of machine learning models to predict danceability by using some of the features provided by Spotify.

#### Deliverables
1. A final notebook. This notebook will be used as a walkthough to present results and allow users to recreate the findings.
2. Python modules that automate the data pipeline process. These modules will be imported and used in the final notebook.
3. This README that explains what the project is, how to reproduce the work, and any notes from project planning.
4. A Trello board that details the process of creating this project.


## Executive Summary 
[(Back to top)](#table-of-contents)

My best models Were the Polynomial Regressor (Degree 2) and the K Nearest Neighbor Regressor. I ultimately chose the KNN Regressor since it performed slightly better then the Polynomial Regressor.

Metrics for the above features:

train metrics: RMSE = 0.53334, R^2 = 0.71555
\
validate metrics: RMSE = 0.60032, R^2 = 0.55669
\
test_error: RMSE = 0.65299, R^2 = 0.69091

---
- No one feature was the end all be all of drivers, but some were certainly bigger drivers than others
- Despite my intial thoughts, popularity was not the biggest driver of danceability
- The best performing models were K-Nearest Neighbor and Polynomial Regressor
    - KNN Regressor did slightly better and reasonably predicted danceability


#### Next Steps

Given more time, I would've wanted to:
- create clusters to increase the accuaracy of my models
- find genres of the songs and include that as a feature
- see if the number of features on a song help predict danceability
- determine if the number of "unique" words spoken could act as a predictor


## Data Dictionary
[(Back to top)](#table-of-contents)


key|datatype|description
|:------------------|:------------------------|:-------------|                   
Unnamed: 0 (*Dropped*)    |int64              |Leftover index from original csv  |
artist_name (*Dropped*)   |object             |Name of the artist |
track_name (*Dropped*)    |object             |Name of the song |
track_id (*Dropped*)      |object             |Area of the lot in square feet |
popularity                |int64              |Quality of build of property |
danceability              |float64            |How suitable a track is for dancing based on a combination of musical elements |
energy                    |float64            |Perceptual measure of intensity and activity. |
key                       |int64              |The key the track is in |
loudness                  |float64            |The overall loudness of a track in decibels (dB)  |
mode                      |int64              |The modality (major or minor) of a track |
speechiness               |float64            |The presence of spoken words in a track |
acousticness              |float64            |A confidence measure from 0.0 to 1.0 of whether the track is acoustic |
instrumentalness          |float64            |Whether a track contains no vocals (Higher value = less vocals) |
liveness                  |float64            |Detects the presence of an audience in the recording |
valence                   |float64            |A measure describing the musical positiveness conveyed by a track |
tempo                     |float64            |The overall estimated tempo of a track in beats per minute (BPM) |
duration_ms               |int64              |The duration of the track in milliseconds |
time_signature            |int64              |Meter of a song (number of beats in a measure) |
name_length               |int64              |Number of characters in the artist_name feature |
song_name_length          |int64              |Number of characters in the track_name feature |
minutes                   |category           |Categorized duration of the track in minutes|
(0, 1] *New*              |uint8              |Duration of the song from 0 minutes up to and including 1 minute |
(1, 2] *New*              |uint8              |Duration of the song from 1 minute up to and including 2 minutes |
(2, 3] *New*              |uint8              |Duration of the song from 2 minute up to and including 3 minutes |
(3, 4] *New*              |uint8              |Duration of the song from 3 minute up to and including 4 minutes |
(4, 5] *New*              |uint8              |Duration of the song from 4 minute up to and including 5 minutes |
(5, 25] *New*             |uint8              |Duration of the song from 5 minute up to and including 25 minutes |
 
 
## Pipeline
[(Back to top)](#table-of-contents)

### Plan
- Build this README and [Trello board](https://trello.com/b/iETQYstZ/spotify-danceability) and continue to add on to these during each phase
- Save the 'world_data.csv' created by [rafaelnduarte](https://www.kaggle.com/rafaelnduarte/spotify-data-with-audio-features) using [Spotify's Web API](https://developer.spotify.com/documentation/web-api/)
- Plan stages of the pipeline and record initial thoughts about the data
- Create hypotheses to test
- Begin next phase

### Acquire
- Create a pandas dataframe using the saved csv
- The data was already cleaned when I got it so I didnt need to change datatypes or drop nulls
- I wasn't sure how to use track id, track name, or artist name as features
- Begin next phase

### Prepare
- I wanted to keep some form of artist_name and track_name so I recorded the number of characters for each column
- I wanted to bin duration_ms so I made the minutes column and dummy columns for modeling
- I dropped the old index column (Unnamed: 0), artist_name, track_name and track_id because I had no further use for them
- I plotted histograms for the final dataframe
- I then split the data to train, validate, and test

### Explore
- I made some quick visualizations (univariate, biivariate, multiivariate)
- I explored and answered my hypotheses using statistical methods
- I summaraized my findings and prepared for the modeling phase

### Model
- Create baseline model using mean to compare to future models
- Using Recursive Feature Elimination and Select KBest, I found the best features to use and decided on the following features:
    - popularity
    - energy
    - loudness
    - speechiness
    - acousticness
    - instrumentalness
    - valence
    - tempo
    - time_signature
    - name_length
    - song_name_length
    - (2, 3]
    - (5, 25]
    
- Models used:
    - OLS
    - LassoLars
    - Polynomial Regressor
    - K-Nearest Neighbor
- I visualized the results of the models and their predictions
- Calculated metrics for each model aid comparison
- Choose K-Nearest Neighbor best model to evaluate with test dataset 
-Best Model Parameters:
    - n_neighbors=8
    - weights='uniform'
    - algorithm='kd_tree'
    - leaf_size=20


## Conclusions
    
- Most characteristics of danceability are correlated but none were strongly correlated enough to single-handedly predict danceability
- I was a little surprised that popularity did not appear to be the biggest driver of danceability
- Loudness was the biggest driver of danceability but not by much
- Artist name length and song name length was had was surpisngly stronger predictor than anticipated

#### Next Steps

Given more time, I would've wanted to:
- create clusters to increase the accuaracy of my models
- find genres of the songs and include that as a feature
- see if the number of features on a song help predict danceability
- determine if the number of "unique" words spoken could act as a predictor


# Recreate these results
[(Back to top)](#table-of-contents)

1. Download this [README](https://github.com/Jason-Tellez/danceability/blob/main/README.md)
2. Download the modules
            [data_wrangle](https://github.com/Jason-Tellez/danceability/blob/main/data_wrangle.py), 
            [explore](https://github.com/Jason-Tellez/danceability/blob/main/explore.py),
            [model](https://github.com/Jason-Tellez/danceability/blob/main/model.py)
            
3. Download the ['world_data.csv' I chose from Kaggle](https://www.kaggle.com/rafaelnduarte/spotify-data-with-audio-features) or create your own using [Spotify's Web API](https://developer.spotify.com/documentation/web-api/)
4. Download [the final jupyter notebook](https://github.com/Jason-Tellez/danceability/blob/main/final_nb.ipynb) and run all cells


