import os
from numpy import true_divide 
import pandas as pd
from pandas.core.frame import DataFrame
import csv
from pandas.io.parsers import read_csv

import bioinfokit
from bioinfokit.analys import get_data,stat



'''
username = "唐"
users_path = "users.csv"
if not os.path.exists(users_path):
    user_df = DataFrame(
        data=[[username,'','','','']],
        columns=("Username","Recommend_Algo","Round_of_Recommendation","Movie_id","Rate_of_user"))
    user_df.to_csv(users_path,index=True)
# check same name
users_df = read_csv(users_path,index_col=0,header=0)
# print(users_df)
# print(users_df["Username"].values)
if username in users_df["Username"].values:
    print("false")
else:
    users_df = users_df.append([{"Username":username}], ignore_index=False)
    users_df = users_df.reset_index(drop=True)
    # print(users_df)
    users_df.to_csv(users_path,index=True)
    # print("true")
'''