#!/usr/bin/env python
#coding=utf-8
import pandas as pd
import re
import pickle
import os
import warnings
import warnings
warnings.filterwarnings("ignore")

def preprocessing_users():
    # user数据处理
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table(r'E:\dl_re_web\re_sys\recommend\data\users.dat', sep='::', header=None, names=users_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    # 改变 User的性别
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)
    # 改变 User的年龄
    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)
    return users,users_orig


def preprocessing_movies():
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('E:\dl_re_web\re_sys\recommend\data\movies.dat', sep='::', header=None, names=movies_title, engine='python')
    movies_orig = movies.values
    # 将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)
    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

    movies['Genres'] = movies['Genres'].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

    movies['Title'] = movies['Title'].map(title_map)
    return movies,title_count,title_set,genres2int,movies_orig

def preprocessing_ratings():
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('E:\dl_re_web\re_sys\recommend\data\ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    return ratings

def load_dataset(model_path):
    """
    title_count：Title字段的长度（15）
    title_set：Title文本的集合
    genres2int：电影类型转数字的字典
    features：是输入X
    targets_values：是学习目标y
    ratings：评分数据集的Pandas对象
    users：用户数据集的Pandas对象
    movies：电影数据的Pandas对象
    data：三个数据集组合在一起的Pandas对象
    movies_orig：没有做数据处理的原始电影数据
    users_orig：没有做数据处理的原始用户数据
    :return:
    """
    print('starting load data ...')

    if  os.path.exists(model_path):
        print('model %s 存在' % model_path)
        # 以二进制文件读取
        return pickle.load(open(model_path, "rb"))

    users,users_orig = preprocessing_users()
    movies,title_count,title_set,genres2int,movies_orig = preprocessing_movies()
    ratings = preprocessing_ratings()
    # 合并三张表
    data = pd.merge(pd.merge(ratings, users), movies)
    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    features = features_pd.values
    targets_values = targets_pd.values

    # 将数据存入模型
    pickle.dump((
                title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig,
                users_orig), open('E:\dl_re_web\re_sys\recommend\model\data_preprocess.pkl', 'wb'))
    print('load data success.')
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


if __name__ == '__main__':
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_dataset('model/data_preprocess.pkl')
    print(movies_orig[0][1])

    # print(movies_orig.shape)
    # print(features.take(0, 1))
    # print(len(features))

