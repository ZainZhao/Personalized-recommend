#!/usr/bin/env python
#coding=utf-8
from django.shortcuts import render
from re_sys.recommend import re_model
from re_sys.recommend import utils
import time

print('----初始化加载模型----')
global_model = re_model.Model()
global_loaded_graph, global_sess = global_model.loead_sess()

# Create your views here.
def index(request):

    return render(request,'index.html')


def recommend(request):
    movie_id = request.GET.get('movie_id')
    try:
        if((int(movie_id)<0) | (int(movie_id)>3953)):
            return render(request,'index.html')
    except ValueError:
        return render(request, 'index.html')

    global global_model
    model=global_model
    print('-------正在推荐--------',movie_id)


    global_loaded_graph, global_sess


    choice_movie_name,list_same_movies_names,list_pepole_like_movies_names,list_same_movies_ids,list_pepole_like_movies_ids =model.recommend_by_movie(int(movie_id))


    print('选择电影：',choice_movie_name)
    print('相似的电影：',list_same_movies_names)
    print('喜欢这个电影的人还喜欢：',list_pepole_like_movies_names)

    list_dict_choice=[]
    for i in choice_movie_name:
        # time.sleep(0.2)  # 爬虫速度
        list_dict_choice.append(utils.movie_dic(i))
    list_dict_choice[0]['movie_id']=movie_id
    # list_dict_choice[0]['title']=choice_movie_name



    list_dict_same = []
    for i in list_same_movies_names[:4]:
        # time.sleep(0.2)
        list_dict_same.append(utils.movie_dic(i))
    for i in range(len(list_dict_same)):
        list_dict_same[i]['movie_id']=list_same_movies_ids[i]


    list_dict_otherlike = []
    for i in list_pepole_like_movies_names[:4]:
        # time.sleep(0.2)
        list_dict_otherlike.append(utils.movie_dic(i))
    for i in range(len(list_dict_otherlike)):
        list_dict_otherlike[i]['movie_id'] = list_pepole_like_movies_ids[i]
        #list_dict_otherlike[i]['title'] = list_dict_otherlike[i]



    print('返回结果')
    print(list_dict_choice)
    print(len(list_dict_same))
    # print(len(list_dict_otherlike))
    context = {}
    context['list_dict_choice'] = list_dict_choice[:4]
    context['list_dict_same'] = list_dict_same
    context['list_dict_otherlike'] = list_dict_otherlike

    return render(request,'index.html',context)

