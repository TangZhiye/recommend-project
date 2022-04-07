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
import time

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
    return {'genre':['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
     'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
     'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western', 'Other']}


@app.post("/api/login")
def login(username: list):
    username = username[0]
    print(username)
    users_path = "users.csv"
    # check same name
    if os.path.exists(users_path):
        users_df = read_csv(users_path,index_col=0,header=0)
        print(users_df)
        if username in users_df["Username"].values:
            return {"result": False}
    return {"result": True}


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
    username = movies[0]
    movies = movies[1]
    print(movies)
    print('username!!!')
    print(username)
    iid = str(sorted(movies, key=lambda i: i['score'], reverse=True)[0]['movie_id'])
    score = int(sorted(movies, key=lambda i: i['score'], reverse=True)[0]['score'])
    res = get_initial_items(iid,score)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'feedback'] = None
    # results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_year', 'poster_url', 'feedback']]
    return json.loads(results.to_json(orient="records"))

@app.post("/api/fisrt_feedback")
def store_first_feedback(first_feedback: list):
    username = first_feedback[0]
    first_feedback = first_feedback[1]
    print(username)
    print(first_feedback)
    users_path = "users.csv"
    # initialize users.csv, add first user
    data = []
    if not os.path.exists(users_path):
        for movie_id, rate in first_feedback.items():
            data.append([1,username, '', '1st_round', movie_id, rate]) 
        new_user_df = DataFrame(
                data=data,
                columns=("User_id","Username","Recommend_Algo","Round_of_Recommendation","Movie_id","Rate_of_user"))
        new_user_df.to_csv(users_path,index=True)
        return {"result": True}
    # store username
    users_df = read_csv(users_path,index_col=0,header=0)
    # print(users_df)
    print(users_df.iloc[-1,0])
    new_user_id = users_df.iloc[-1,0] + 1
    for movie_id, rate in first_feedback.items():
            data.append([new_user_id,username, '', '1st_round', movie_id, rate]) 
    new_user_df = DataFrame(
            data=data,
            columns=("User_id","Username","Recommend_Algo","Round_of_Recommendation","Movie_id","Rate_of_user"))
    users_df = users_df.append(new_user_df, ignore_index=False)
    users_df = users_df.reset_index(drop=True)
    users_df.to_csv(users_path,index=True)
    # results = get_second_recommend(first_feedback)
    return {"result": True}
    # return json.loads(results.to_json(orient="records"))

@app.post("/api/add_recommend")
async def add_recommend(first_feedback: list):
# def get_second_recommend(first_feedback: list):
    print('2nd recommend start!')
    username = first_feedback[0]
    first_feedback = first_feedback[1]  
    # 以下是临时代码，用来继续搭建第二轮推荐的前端
    item_id = list(first_feedback.keys())[0]
    res = get_similar_items(str(item_id), n=12)
    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'feedback'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_year', 'poster_url', 'feedback']]
    return json.loads(results.to_json(orient="records"))
    # return results

def user_add(iid, score):
    user = '611'
    # simulate adding a new user into the original data file
    # df = pd.read_csv('./u.data')
    # df.to_csv('new_' + 'u.data')
    df = pd.read_csv('./ml-latest-small/ratings.csv',header=0)
    df.to_csv('new_' + 'ratings.csv',header=False,index=False)
    with open(r'new_ratings.csv',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter=',')
        data_input = []
        s = [user,str(iid),int(score),int(time.time())]
        data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def get_initial_items(iid, score, n=12):
    res = []
    user_add(iid, score)
    # file_path = os.path.expanduser('new_ratings.csv')
    file_path = './new_ratings.csv'
    reader = Reader(line_format='user item rating timestamp', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in range(9742):
        uid = str(611)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res

def get_similar_items(iid, n=12):
    algo = dump.load('./model')[1]
    inner_id = algo.trainset.to_inner_iid(iid)
    print(inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid
