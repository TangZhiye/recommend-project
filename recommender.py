from surprise import SVD
from surprise import Dataset
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import dump
from surprise import Reader
from surprise.model_selection import cross_validate
import math
import numpy as np
from scipy import spatial
from scipy import stats
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import NormalPredictor
from surprise import accuracy
from surprise import Dataset
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
import os
from surprise import Reader
from typing import Optional, List

from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split

from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset
from random import choice
#====================================================
from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset
import time
import timeit
from surprise import SVDpp

def select_k():
    file_path_pre = './ml-latest-small/ratings.csv'

    data_df = pd.read_csv(file_path_pre, names=['userId', 'movieId', 'rating', 'timestamp'], sep=',', header=0)
    reader = Reader(rating_scale=(1, 5))
    data_pre = Dataset.load_from_df(data_df[['userId', 'movieId', 'rating']], reader)

    train_set, test_set = train_test_split(data_pre, test_size=.25)
    k_list = [20, 40, 60, 80, 125, 200]
    centeredKNN_Pearson_rmse_user = []
    for k in k_list:
        algo1 = KNNWithMeans(k, sim_options={
            "name": "pearson",
            "user_based": True,
        })
        algo1.fit(train_set)
        predictions = algo1.test(test_set)
        centeredKNN_Pearson_rmse_user.append(accuracy.rmse(predictions, verbose=True))
    k_Method_1 = k_list[centeredKNN_Pearson_rmse_user.index(min(centeredKNN_Pearson_rmse_user))]
    return k_Method_1

def Method_2(n):
    # content-based movie recommender with movie genres
    movies_data = './/ml-latest-small//movies.csv'
    ratings_data = 'new_ratings.csv'
    movies_df = pd.read_csv(movies_data)
    ratings_df = pd.read_csv(ratings_data, header=None, names=["user_id", "movie_id", "rating", "timestamp"])
    # Every genre is separated by a | so we simply have to call the split function on |
    movies_df['genres'] = movies_df.genres.str.split('|')
    # First let's make a copy of the movies_df
    movies_with_genres = movies_df[['movieId', 'title', 'genres']].copy(deep=True)
    # Let's iterate through movies_df, then append the movie genres as columns of 1s or 0s.
    # 1 if that column contains movies in the genre at the present index and 0 if not.
    genre_list = []  # store the occurred genres
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)
    # Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    movies_with_genres = movies_with_genres.fillna(0)
    movies_with_genres.rename(columns={'movieId': 'movie_id'}, inplace=True)
    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()
    #new user profile build
    ratings_df = pd.read_csv(ratings_data, header=None, names=["user_id", "movie_id", "rating", "timestamp"])
    new_user_rating_df = ratings_df[ratings_df['user_id'] == 611]
    new_user_rating_df = new_user_rating_df.reset_index(drop=True)
    user_movie_rating_df = pd.merge(new_user_rating_df, movies_with_genres)
    user_movie_df = user_movie_rating_df.copy(deep=True)
    # Next, let's maintain the genre-related information
    user_movie_df = user_movie_df[genre_list]
    rating_weight = new_user_rating_df.rating / new_user_rating_df.rating.sum()
    user_profile = user_movie_df.T.dot(rating_weight)
    #finish build user profile
    movies_with_genres = movies_with_genres[genre_list]
    #nomalize
    user_profile_normalized = user_profile / sum(user_profile.values)
    user_profile_normalized.sort_values()
    # Compute the cosine similarity
    u_v = user_profile_normalized.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, movies_genre_matrix)
    recommendation_table_df = movies_df[['movieId', 'title']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    # We can get the recommendation results by sorting the movies using similarity
    newuser_rated_list = new_user_rating_df["movie_id"]
    rec_result_GENRE_based = recommendation_table_df.sort_values(by=['similarity'], ascending=False)
    for i in newuser_rated_list:
        rec_result_GENRE_based = rec_result_GENRE_based[~rec_result_GENRE_based['movieId'].isin([i])]
    res_store = rec_result_GENRE_based.head(n)
    res_int =res_store["movieId"].tolist()
    res = [str(x) for x in res_int]
    return res

def Method_1(k_Method_1,n):
    res=[]
    file_path = 'new_ratings.csv'
    reader = Reader(line_format='user item rating timestamp', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo= KNNWithMeans(k_Method_1, sim_options={
        "name": "pearson",
        "user_based": True,
    })
    algo.fit(trainset)
    dump.dump('./model', algo=algo, verbose=1)
    all_results = {}
    new_data = pd.read_csv('new_ratings.csv',header=None, names=["user_id", "movie_id", "rating", "timestamp"])
    new_data_df = new_data[new_data['user_id'] == 611]
    print('new_data_df!!!!!!!!!')
    print(new_data_df)
    iid_list = new_data_df['movie_id'].values.tolist()
    # iid_list 第二轮和第一轮一样
    print('iid_list!!!!!!')
    print(iid_list)
    i_select = pd.read_csv('movie_info_latest.csv')
    i_list = i_select['movie_id'].values.tolist()
    for i in i_list:
        if i not in iid_list:
            uid = str(611)
            iid = str(i)
            pred = algo.predict(uid, iid).est
            all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res

