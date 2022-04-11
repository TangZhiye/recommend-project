import pandas as pd
from collections import defaultdict
import numpy as np
from bioinfokit.analys import get_data,stat
from scipy import stats as st
# from bioinfokit import analys,visuz

class Evaluation():
    def __init__(self):
        self.dic1_round1 = {}      #第一轮方法一所有用户打分，字典
        self.dic1_round2 = {}      #第二轮方法一所有用户打分，字典
        self.dic2_round1 = {}      #第一轮方法二所有用户打分，字典
        self.dic2_round2 = {}      #第二轮方法二所有用户打分，字典
        self.path = "./Demo.csv"
        self.user_average_score_list_round1_algo1 = []      #第一轮方法一所有分数集合
        self.user_average_score_list_round2_algo1 = []      #第二轮方法一所有分数集合
        self.user_average_score_list_round1_algo2 = []      #第一轮方法二所有分数集合
        self.user_average_score_list_round2_algo2 = []      #第二轮方法二所有分数集合

    def Read_csv(self):

        df = pd.read_csv(self.path)
        self.dic1_round1 = defaultdict(list)        #创建字典形式{用户：评分}
        self.dic1_round2 = defaultdict(list)
        self.dic2_round1 = defaultdict(list)
        self.dic2_round2 = defaultdict(list)
        for index in range(len(df)):
            if df['Recommend_Algo'][index] == "Method_1":        #分情况添加进字典
                if df['Round_of_Recommendation'][index] == "1st_round":       #方法一第一轮情况
                    self.dic1_round1[df['User_id'][index]].append(df['Rate_of_user'][index])
                elif df['Round_of_Recommendation'][index] == "2nd_round":
                    self.dic1_round2[df['User_id'][index]].append(df['Rate_of_user'][index])
            elif df['Recommend_Algo'][index] == "Method_2":
                if df['Round_of_Recommendation'][index] == "1st_round":
                    self.dic2_round1[df['User_id'][index]].append(df['Rate_of_user'][index])
                elif df['Round_of_Recommendation'][index] == "2nd_round":
                    self.dic2_round2[df['User_id'][index]].append(df['Rate_of_user'][index])
        print(self.dic1_round1)
        for key in self.dic1_round1:        #算每个字典平均分值然后存入列表
            average_score = np.mean(self.dic1_round1[key])
            self.user_average_score_list_round1_algo1.append(average_score)
        for key in self.dic1_round2:
            average_score = np.mean(self.dic1_round2[key])
            self.user_average_score_list_round2_algo1.append(average_score)
        for key in self.dic2_round1:
            average_score = np.mean(self.dic2_round1[key])
            self.user_average_score_list_round1_algo2.append(average_score)
        for key in self.dic2_round2:
            average_score = np.mean(self.dic2_round2[key])
            self.user_average_score_list_round2_algo2.append(average_score)

    def Paired_test(self):          #实现目标Paired t-Test
        data = []       #第一轮
        print(self.user_average_score_list_round1_algo1,self.user_average_score_list_round2_algo1)
        data.append(self.user_average_score_list_round1_algo1)      #合并两个方法的列表数据
        data.append(self.user_average_score_list_round2_algo1)
        arr = list(map(list, zip(*data)))       #倒转矩阵
        df = pd.DataFrame(arr, columns=['AF', 'BF'])
        print(df)
        res = stat()
        res.ttest(df=df,res=['AF','BF'],test_type=3)
        print(res.summary)

        data = []       #第二轮
        data.append(self.user_average_score_list_round1_algo2)      #合并两个方法的列表数据
        data.append(self.user_average_score_list_round2_algo2)
        arr = list(map(list, zip(*data)))       #倒转矩阵
        df = pd.DataFrame(arr, columns=['AF', 'BF'])
        print(df)
        res = stat()
        res.ttest(df=df, res=['AF', 'BF'], test_type=3)
        print(res.summary)

    def Two_sample_test(self):
        score = []      #第一轮
        data = []
        name = ''
        score = self.user_average_score_list_round1_algo1+self.user_average_score_list_round1_algo2     #合并分数列表
        len1 = len(self.user_average_score_list_round1_algo1)
        len2 = len(self.user_average_score_list_round1_algo2)
        name = 'Method_1'*len1 +'Method_2'*len2       #填充方法便于构建dataframe
        namelist = list(name)
        data.append(namelist)
        data.append(score)
        arr = list(map(list, zip(*data)))       #倒转矩阵
        df = pd.DataFrame(arr, columns=['Genotype', 'yield'])
        print(df)
        a = df.loc[df['Genotype'] == 'Method_1','yield'].to_numpy()
        b = df.loc[df['Genotype'] == 'Method_2', 'yield'].to_numpy()
        print(st.ttest_ind(a=a,b=b,equal_var=True))

        score = []      #第二轮
        data = []
        name = ''
        score = self.user_average_score_list_round2_algo1 + self.user_average_score_list_round2_algo2       #合并分数列表
        len1 = len(self.user_average_score_list_round2_algo1)
        len2 = len(self.user_average_score_list_round2_algo2)
        name = 'Method_1' * len1 + 'Method_2' * len2      #填充方法便于构建dataframe
        namelist = list(name)
        data.append(namelist)
        data.append(score)
        arr = list(map(list, zip(*data)))       #倒转矩阵
        df = pd.DataFrame(arr, columns=['Genotype', 'yield'])
        print(df)
        a = df.loc[df['Genotype'] == 'Method_1', 'yield'].to_numpy()
        b = df.loc[df['Genotype'] == 'Method_2', 'yield'].to_numpy()
        print(st.ttest_ind(a=a, b=b, equal_var=True))






if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.Read_csv()
    evaluation.Paired_test()
    evaluation.Two_sample_test()
