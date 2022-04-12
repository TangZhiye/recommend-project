from typing import Optional, List
# ====================================================
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
from recommender import Method_2
from recommender import Method_1
from recommender import select_k
from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset
from random import choice
# ====================================================
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
# using ml-latest-small dataset
data = pd.read_csv("movie_info_latest.csv")
method = ""

k_Method_1 = select_k()
n = 12  # recommend 12 movies for round 1 and 2

"""
=================== Body =============================
"""


class Movie(BaseModel):
    movie_id: int
    movie_title: str
    release_year: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =

# show all generes
# @app.get("/api/genre")
# def get_genre():
#     return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
#                       "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
#                       "Romance", "Sci_Fi", "Thriller", "War", "Western"]}

# ml-latest-small dataset genre
@app.get("/api/genre")
def get_genre():
    return {'genre': ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
                      'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western', 'Other']}


@app.post("/api/login")
def login(username: list):
    global method
    method = ""
    username = username[0]
    print(username)
    users_path = "users.csv"
    users_method = "users_method.csv"
    users_method_df = read_csv(users_method, index_col=0, header=0)
    # check same name
    if os.path.exists(users_path):
        users_df = read_csv(users_path, index_col=0, header=0)
        print(users_df)
        if username in users_df["Username"].values:
            return {"result": False}
        else:
            # give a random algo
            if len(users_method_df[users_method_df["Recommend_Algo"] == "Method_1"]) < 10:
                if len(users_method_df[users_method_df["Recommend_Algo"] == "Method_2"]) < 10:
                    recommendAlgo = choice(['Method_1', 'Method_2'])
                else:
                    recommendAlgo = "Method_1"
            else:
                if len(users_method_df[users_method_df["Recommend_Algo"] == "Method_2"]) < 10:
                    recommendAlgo = "Method_2"
                else:
                    # return {"result": False}
                    recommendAlgo = choice(['Method_1', 'Method_2'])
    else:
        recommendAlgo = choice(['Method_1', 'Method_2'])
    method = recommendAlgo
    print('current user rec method!!!!!!!!!!!')
    print(method)
    new = pd.DataFrame({"User_name": username,
                        "Recommend_Algo": recommendAlgo
                        }, index=[0]
                       )
    users_method_df = users_method_df.append(new, ignore_index=True)
    users_method_df.to_csv(users_method, index=True)
    return {"result": True}

@app.get("/api/method")
def getMethod():
    global method
    print('method!!!!!!!!!!')
    print(method)
    return {"method": method}

@app.post("/api/save_user_info")
def save_user_info(user_info: list):
    print(user_info)
    with open(r'users_info.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter=',')
        wf.writerow(user_info)

@app.post("/api/movies")
def get_movies(genre: list):
    print(genre)
    query_str = " or ".join(map(map_genre, genre))
    print(query_str)
    results = data.query(query_str)
    # print(results)
    results.loc[:, 'score'] = None
    results = results.sample(18).loc[:, ['movie_id', 'movie_title', 'release_year', 'poster_url', 'score']]
    # print(results.to_json(orient="records"))
    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend")
def get_recommend(movies: list):
    # username = movies[0]
    movies = movies[1]
    print(movies)
    user_add(movies)
    res = recommend()
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print("this is res!!!!!!!!!!!!!!")
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'feedback'] = None
    # results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_year', 'poster_url', 'feedback']]
    return json.loads(results.to_json(orient="records"))


@app.post("/api/fisrt_feedback")
def store_first_feedback(first_feedback: list):
    print("this is first feedback!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(first_feedback)
    username = first_feedback[0]
    first_feedback = first_feedback[1]
    print(username)
    print("this is the type of feedback")
    print(type(first_feedback))
    users_path = "users.csv"
    # initialize users.csv, add first user
    data = []
    # rating = []
    if not os.path.exists(users_path):
        for movie_id, rate in first_feedback.items():
            data.append([1, username, method, '1st_round', movie_id, rate])
        new_user_df = DataFrame(
            data=data,
            columns=("User_id", "Username", "Recommend_Algo", "Round_of_Recommendation", "Movie_id", "Rate_of_user"))
        print('new_user_df111111111111111111111111!!!!!!!')
        print(new_user_df)
        new_user_df.to_csv(users_path, index=True)
    else:
        # store username
        users_df = read_csv(users_path, index_col=0, header=0)
        # print(users_df)
        print(users_df.iloc[-1, 0])
        new_user_id = users_df.iloc[-1, 0] + 1
        for movie_id, rate in first_feedback.items():
            data.append([new_user_id, username, method, '1st_round', movie_id, rate])
            # rating.append([611,movie_id,rate,int(time.time())])
        new_user_df = DataFrame(
            data=data,
            columns=("User_id", "Username", "Recommend_Algo", "Round_of_Recommendation", "Movie_id", "Rate_of_user"))
        print('new_user_df22222222222222222222!!!!!!!')
        print(new_user_df)
        # store new_user_df to users.csv
        users_df = users_df.append(new_user_df, ignore_index=False)
        users_df = users_df.reset_index(drop=True)
        print("111111111111111111111111111111")
        users_df.to_csv(users_path, index=True)
    # add new users first feedback to new_rating.csv
    print("22222222222222222222222222222222")
    add_feedback(first_feedback)

    return {"result": True}


@app.post("/api/second_feedback")
def store_second_feedback(second_feedback: list):
    print("this is second feedback!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(second_feedback)
    username = second_feedback[0]
    second_feedback = second_feedback[1]
    users_path = "users.csv"
    data = []
    # store username
    users_df = read_csv(users_path, index_col=0, header=0)
    # print(users_df)
    print(users_df.iloc[-1, 0])
    new_user_id = users_df.iloc[-1, 0] 
    for movie_id, rate in second_feedback.items():
        data.append([new_user_id, username, method, '2nd_round', movie_id, rate])
    new_user_df = DataFrame(
        data=data,
        columns=("User_id", "Username", "Recommend_Algo", "Round_of_Recommendation", "Movie_id", "Rate_of_user"))
    print('new_user_df!!!!!!!')
    print(new_user_df)
    # store new_user_df to users.csv
    users_df = users_df.append(new_user_df, ignore_index=False)
    users_df = users_df.reset_index(drop=True)
    users_df.to_csv(users_path, index=True)
    return {"result": True}


@app.post("/api/add_recommend")
# async def add_recommend(first_feedback: list):
def add_recommend(first_feedback: list):
    # time.sleep(1)
    print('2nd recommend start!')
    username = first_feedback[0]
    first_feedback = first_feedback[1]
    item_id = list(first_feedback.keys())[0]
    res = recommend()
    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'feedback'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_year', 'poster_url', 'feedback']]
    return json.loads(results.to_json(orient="records"))
    # return results


def user_add(movies):
    user = '611'
    # simulate adding a new user into the original data file
    # df = pd.read_csv('./u.data')
    # df.to_csv('new_' + 'u.data')
    df = pd.read_csv('./ml-latest-small/ratings.csv', header=0)
    df.to_csv('new_' + 'ratings.csv', header=False, index=False)
    with open(r'new_ratings.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter=',')
        data_input = []
        for m in range(len(movies)):
            iid_m = str(sorted(movies, key=lambda i: i['score'], reverse=True)[m]['movie_id'])
            print(iid_m)
            score_m = int(sorted(movies, key=lambda i: i['score'], reverse=True)[m]['score'])
            s = [user, str(iid_m), int(score_m), int(time.time())]
            data_input.append(s)
        print("this is data_input!!!!!!!!!!!!!!")
        print(data_input)
        for k in data_input:
            wf.writerow(k)


# add first feedback to new_ratings.csv
def add_feedback(first_feedback):
    print("add feedback start")
    user = '611'
    # file_path='new_ratings.csv'
    # rating = read_csv(file_path,header=0)
    # for movie_id,rate in first_feedback.items():
    #     s = [user, str(movie_id), int(rate), int(time.time())]
    #     print("this is movie id")
    #     print(movie_id)
    #     rating.append(s)
    # print(rating)
    # rating.to_csv(file_path, header=False, index=False)
    s = []
    for movie_id, rate in first_feedback.items():
        s.append([user, str(movie_id), int(rate), int(time.time())])
    with open('new_ratings.csv', 'a',newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerows(s)
    print("write into rating finish")


    # with open(r'new_ratings.csv', mode='a', newline='', encoding='utf8') as cfa:
    #     wf = csv.writer(cfa, delimiter=',')
    #     data_input = []
    #     for movie_id, rate in first_feedback.items():
    #         s = [user, str(movie_id), int(rate), int(time.time())]
    #         data_input.append(s)
    #     print("this is first/second feedback data_input!!!!!!!!!!!!!!")
    #     print(data_input)
    #     for k in data_input:
    #         wf.writerow(k)


def recommend():
    # time.sleep(1)
    if method == "Method_1":
        res = Method_1(k_Method_1, n)
    else:
        if method == "Method_2":
            res = Method_2(n)
        else:
            return {"result": False}
    return res

