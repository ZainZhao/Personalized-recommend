#!/usr/bin/env python
#coding=utf-8
from re_sys.recommend import DataSet
import tensorflow as tf
import time
import datetime
import os
import numpy as np
from sklearn.model_selection import train_test_split
from re_sys.recommend import  utils
import pickle
import re

class Model:

    def __init__(self,has_fit=True):
        self.has_fit=has_fit
        # print('初始化model...')
        tf.reset_default_graph()
        self.train_graph = tf.Graph()
        self.title_count, self.title_set, self.genres2int, self.features, self.targets_values, self.ratings, self.users, self.movies, \
        self.data, self.movies_orig, self.users_orig = DataSet.load_dataset(r'E:\dl_re_web\re_sys\recommend\model\data_preprocess.pkl')
        # 嵌入矩阵的维度
        self.embed_dim = 32
        # 用户ID个数
        self.uid_max = max(self.features.take(0, 1)) + 1  # 6040+1   第0列最大值 +1  id编号[1,6040]
        # 性别个数
        self.gender_max = max(self.features.take(2, 1)) + 1  # 1 + 1 = 2
        # 年龄类别个数
        self.age_max = max(self.features.take(3, 1)) + 1  # 6 + 1 = 7
        # 职业个数
        self.job_max = max(self.features.take(4, 1)) + 1  # 20 + 1 = 21

        # 电影ID个数
        self.movie_id_max = max(self.features.take(1, 1)) + 1  # 3952
        # 电影类型个数
        self.movie_categories_max = max(self.genres2int.values()) + 1  # 18 + 1 = 19
        # 电影名单词个数
        self.movie_title_max = len(self.title_set)  # 5216

        # 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
        self.combiner = "sum"

        # 电影名长度
        self.sentences_size = self.title_count  # = 15
        # 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
        self.window_sizes = {2, 3, 4, 5}
        # 文本卷积核数量
        self.filter_num = 8

        # 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
        self.movieid2idx = {val[0]: i for i, val in enumerate(self.movies.values)}

        # 超参
        # Number of Epochs
        self.num_epochs = 5
        # Batch Size
        self.batch_size = 256

        self.dropout_keep_prob = 0.5
        # Learning Rate
        self.learning_rate = 0.0001
        # Show stats for every n number of batches
        self.show_every_n_batches = 20

        self. losses = {'train': [], 'test': []}
        # self.save_dir = './model'
        # 初始化模型
        self.users_matrics=[]
        self.movies_matrics=[]


        if(self.has_fit==True):
            print('初始化：model.meta')
            self.users_matrics=pickle.load(open(r'E:\dl_re_web\re_sys\recommend\model\users_matrics.pkl', mode='rb'))
            self.movies_matrics=pickle.load(open(r'E:\dl_re_web\re_sys\recommend\model\movies_matrics.pkl', mode='rb'))
            # self.loader=tf.train.import_meta_graph(r'E:\dl_re_web\re_sys\recommend\model\model.meta')


    def create_user_embedding(self, uid, user_gender, user_age, user_job):
        """
        构造user的嵌入矩阵
        通的one-hot编码很难表示两个词之间的相关度，但通过可训练的embedding层可以学习出两个词变量编码，且如果是相关的词，词向量之间具有更大的相关性。
        :return:
        """
        with tf.name_scope("user_embedding"):
            uid_embed_matrix = tf.Variable(tf.random_uniform([self.uid_max, self.embed_dim], -1, 1),
                                           name="uid_embed_matrix")# (6041,32)
            uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

            gender_embed_matrix = tf.Variable(tf.random_uniform([self.gender_max, self.embed_dim // 2], -1, 1),
                                              name="gender_embed_matrix")# (2,16)
            gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

            age_embed_matrix = tf.Variable(tf.random_uniform([self.age_max, self.embed_dim // 2], -1, 1),
                                           name="age_embed_matrix")# (7,16)
            age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")# (6041,32)

            job_embed_matrix = tf.Variable(tf.random_uniform([self.job_max, self.embed_dim // 2], -1, 1),
                                           name="job_embed_matrix")# (21,16)
            job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")# (6041,32)
        return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

    def create_user_feature_layer(self, uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
        """
        将User的嵌入矩阵一起全连接生成User的特征
        :return: 
        """
        with tf.name_scope("user_fc"):
            # 第一层全连接
            uid_fc_layer = tf.layers.dense(uid_embed_layer, self.embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
            gender_fc_layer = tf.layers.dense(gender_embed_layer, self.embed_dim, name="gender_fc_layer",
                                              activation=tf.nn.relu)
            age_fc_layer = tf.layers.dense(age_embed_layer, self.embed_dim, name="age_fc_layer", activation=tf.nn.relu)
            job_fc_layer = tf.layers.dense(job_embed_layer, self.embed_dim, name="job_fc_layer", activation=tf.nn.relu)

            # 第二层全连接
            user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer],
                                           2)  # (?, 1, 128)
            user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)
            user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
        return user_combine_layer, user_combine_layer_flat

    def create_movie_id_embed_layer(self, movie_id):
        """
        定义Movie ID 的嵌入矩阵
        :param movie_id:
        :return:
        """
        with tf.name_scope("movie_embedding"):
            movie_id_embed_matrix = tf.Variable(tf.random_uniform([self.movie_id_max, self.embed_dim], -1, 1),
                                                name="movie_id_embed_matrix") # (3953,32)
            movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
        return movie_id_embed_layer

    def create_movie_categories_embed_layers(self, movie_categories):
        """
        对电影类型的多个嵌入向量做加和
        :param movie_categories:
        :return:
        """
        with tf.name_scope("movie_categories_layers"):
            movie_categories_embed_matrix = tf.Variable(
                tf.random_uniform([self.movie_categories_max, self.embed_dim], -1, 1),
                name="movie_categories_embed_matrix")
            movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                                  name="movie_categories_embed_layer")
            if self.combiner == "sum":
                movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
        return movie_categories_embed_layer

    def create_movie_title_cnn_layer(self, movie_titles):
        # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
        with tf.name_scope("movie_embedding"):
            movie_title_embed_matrix = tf.Variable(tf.random_uniform([self.movie_title_max, self.embed_dim], -1, 1),
                                                   name="movie_title_embed_matrix")# (5215,32)
            movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                             name="movie_title_embed_layer") #shape=(?, 15, 32）
            movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)  #shape=(?, 15, 32, 1) #再最后加上一个维度

        # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
        pool_layer_lst = []
        for window_size in self.window_sizes:
            with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
                filter_weights = tf.Variable(
                    tf.truncated_normal([window_size, self.embed_dim, 1, self.filter_num], stddev=0.1),
                    name="filter_weights")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name="filter_bias")

                conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                          name="conv_layer")
                relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

                maxpool_layer = tf.nn.max_pool(relu_layer, [1, self.sentences_size - window_size + 1, 1, 1],
                                               [1, 1, 1, 1],
                                               padding="VALID", name="maxpool_layer")
                pool_layer_lst.append(maxpool_layer)

        # Dropout层
        with tf.name_scope("pool_dropout"):
            pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
            max_num = len(self.window_sizes) * self.filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

            dropout_layer = tf.nn.dropout(pool_layer_flat, self.dropout_keep_prob, name="dropout_layer")
        return pool_layer_flat, dropout_layer

    def create_movie_feature_layer(self, movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
        """
        将Movie的各个层一起做全连接
        :param movie_id_embed_layer:
        :param movie_categories_embed_layer:
        :param dropout_layer:
        :return:
        """
        with tf.name_scope("movie_fc"):
            # 第一层全连接
            movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, self.embed_dim, name="movie_id_fc_layer",
                                                activation=tf.nn.relu)
            movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, self.embed_dim,
                                                        name="movie_categories_fc_layer", activation=tf.nn.relu)

            # 第二层全连接
            movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer],
                                            2)  # (?, 1, 96)
            movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

            movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
        return movie_combine_layer, movie_combine_layer_flat

    def fit(clf):
        # 构建图
        print(clf.age_max)
        with clf.train_graph.as_default():
            # 获取输入占位符
            uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = utils.place_inputs()
            # 获取User的4个嵌入向量
            uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = clf.create_user_embedding(uid,
                                                                                                              user_gender,
                                                                                                              user_age,
                                                                                                              user_job)
            # 得到用户特征
            user_combine_layer, user_combine_layer_flat = clf.create_user_feature_layer(uid_embed_layer,
                                                                                        gender_embed_layer,
                                                                                        age_embed_layer,
                                                                          job_embed_layer)
            # 获取电影ID的嵌入向量
            movie_id_embed_layer = clf.create_movie_id_embed_layer(movie_id)
            # 获取电影类型的嵌入向量
            movie_categories_embed_layer = clf.create_movie_categories_embed_layers(movie_categories)
            # 获取电影名的特征向量
            pool_layer_flat, dropout_layer = clf.create_movie_title_cnn_layer(movie_titles)
            # 得到电影特征
            movie_combine_layer, movie_combine_layer_flat = clf.create_movie_feature_layer(movie_id_embed_layer,
                                                                                           movie_categories_embed_layer,
                                                                                           dropout_layer)
            # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
            with tf.name_scope("inference"):
                # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
                #         inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
                #         inference = tf.layers.dense(inference_layer, 1,
                #                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                #                                     kernel_regularizer=tf.nn.l2_loss, name="inference")
                # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
                #        inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
                inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
                inference = tf.expand_dims(inference, axis=1)

            with tf.name_scope("loss"):
                # MSE损失，将计算值回归到评分
                cost = tf.losses.mean_squared_error(targets, inference)
                loss = tf.reduce_mean(cost)
            # 优化损失
            #     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(lr)   # 传入学习率
            gradients = optimizer.compute_gradients(loss)  # cost
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)



        with tf.Session(graph=clf.train_graph) as sess:

            # 搜集数据给tensorBoard用
            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in gradients:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')),
                                                         tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Inference summaries
            inference_summary_op = tf.summary.merge([loss_summary])
            inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
            inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for epoch_i in range(clf.num_epochs):

                # 将数据集分成训练集和测试集，随机种子不固定
                train_X, test_X, train_y, test_y = train_test_split(clf.features,
                                                                    clf.targets_values,
                                                                    test_size=0.2,
                                                                    random_state=0)

                train_batches = utils.get_batches(train_X, train_y, clf.batch_size)

                test_batches = utils.get_batches(test_X, test_y, clf.batch_size)

                # 训练的迭代，保存训练损失
                for batch_i in range(len(train_X) // clf.batch_size):
                    x, y = next(train_batches)

                    categories = np.zeros([clf.batch_size, 18])
                    for i in range(clf.batch_size):
                        categories[i] = x.take(6, 1)[i]

                    titles = np.zeros([clf.batch_size, clf.sentences_size])
                    for i in range(clf.batch_size):
                        titles[i] = x.take(5, 1)[i]

                    feed = {
                        uid: np.reshape(x.take(0, 1), [clf.batch_size, 1]),
                        user_gender: np.reshape(x.take(2, 1), [clf.batch_size, 1]),
                        user_age: np.reshape(x.take(3, 1), [clf.batch_size, 1]),
                        user_job: np.reshape(x.take(4, 1), [clf.batch_size, 1]),
                        movie_id: np.reshape(x.take(1, 1), [clf.batch_size, 1]),
                        movie_categories: categories,  # x.take(6,1)
                        movie_titles: titles,  # x.take(5,1)
                        targets: np.reshape(y, [clf.batch_size, 1]),
                        dropout_keep_prob: clf.dropout_keep_prob,  # dropout_keep
                        lr: clf.learning_rate}

                    step, train_loss, summaries, _ = sess.run(
                        [global_step, loss, train_summary_op, train_op],
                        feed)  # cost
                    clf.losses['train'].append(train_loss)
                    train_summary_writer.add_summary(summaries, step)  #

                    # Show every <show_every_n_batches> batches
                    if (epoch_i * (len(train_X) // clf.batch_size) + batch_i) % clf.show_every_n_batches == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            time_str,
                            epoch_i,
                            batch_i,
                            (len(train_X) // clf.batch_size),
                            train_loss))

                # 使用测试数据的迭代
                for batch_i in range(len(test_X) // clf.batch_size):
                    x, y = next(test_batches)

                    categories = np.zeros([clf.batch_size, 18])
                    for i in range(clf.batch_size):
                        categories[i] = x.take(6, 1)[i]

                    titles = np.zeros([clf.batch_size, clf.sentences_size])
                    for i in range(clf.batch_size):
                        titles[i] = x.take(5, 1)[i]

                    feed = {
                        uid: np.reshape(x.take(0, 1), [clf.batch_size, 1]),
                        user_gender: np.reshape(x.take(2, 1), [clf.batch_size, 1]),
                        user_age: np.reshape(x.take(3, 1), [clf.batch_size, 1]),
                        user_job: np.reshape(x.take(4, 1), [clf.batch_size, 1]),
                        movie_id: np.reshape(x.take(1, 1), [clf.batch_size, 1]),
                        movie_categories: categories,  # x.take(6,1)
                        movie_titles: titles,  # x.take(5,1)
                        targets: np.reshape(y, [clf.batch_size, 1]),
                        dropout_keep_prob: 1,
                        lr: clf.learning_rate}

                    step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op],
                                                          feed)  # cost

                    # 保存测试损失
                    clf.losses['test'].append(test_loss)
                    inference_summary_writer.add_summary(summaries, step)  #

                    time_str = datetime.datetime.now().isoformat()
                    if (epoch_i * (len(test_X) // clf.batch_size) + batch_i) % clf.show_every_n_batches == 0:
                        print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                            time_str,
                            epoch_i,
                            batch_i,
                            (len(test_X) // clf.batch_size),
                            test_loss))

            # Save Model
            saver.save(sess, r'E:\dl_re_web\re_sys\recommend\model\model')  # , global_step=epoch_i

            clf.create_movies_matrics()
            clf.create_users_matrics()
            print('Model Trained and Saved')

    def loead_sess(self):
        loaded_graph = tf.Graph()
        sess  =tf.Session(graph=loaded_graph)
        with loaded_graph.as_default():
            print('加载模型')
            loader = tf.train.import_meta_graph(r'E:\dl_re_web\re_sys\recommend\model\model.meta')
            loader.restore(sess, r'E:\dl_re_web\re_sys\recommend\model\model')
        return loaded_graph,sess

    def rating_movie_1(self,sess,loaded_graph,user_id_val,movie_id_val):
        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles,\
        targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = utils.get_tensors(
            loaded_graph)  # loaded_graph

        categories = np.zeros([1, 18])
        categories[0] = self.movies.values[self.movieid2idx[movie_id_val]][2]

        titles = np.zeros([1, self.sentences_size])
        titles[0] = self.movies.values[self.movieid2idx[movie_id_val]][1]

        feed = {
            uid: np.reshape(self.users.values[user_id_val - 1][0], [1, 1]),
            user_gender: np.reshape(self.users.values[user_id_val - 1][1], [1, 1]),
            user_age: np.reshape(self.users.values[user_id_val - 1][2], [1, 1]),
            user_job: np.reshape(self.users.values[user_id_val - 1][3], [1, 1]),
            movie_id: np.reshape(self.movies.values[self.movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1}

        # Get Prediction
        inference_val = sess.run([inference], feed)
        return (inference_val)

    def rating_movie(self,user_id_val,movie_id_val):

        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #
            print('加载模型')

            loader = tf.train.import_meta_graph(r'E:\dl_re_web\re_sys\recommend\model\model.meta')
            loader.restore(sess, r'E:\dl_re_web\re_sys\recommend\model\model')

            # Get Tensors from loaded model
            uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles,\
            targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = utils.get_tensors(
                loaded_graph)  # loaded_graph

            categories = np.zeros([1, 18])
            categories[0] = self.movies.values[self.movieid2idx[movie_id_val]][2]

            titles = np.zeros([1, self.sentences_size])
            titles[0] = self.movies.values[self.movieid2idx[movie_id_val]][1]

            feed = {
                uid: np.reshape(self.users.values[user_id_val - 1][0], [1, 1]),
                user_gender: np.reshape(self.users.values[user_id_val - 1][1], [1, 1]),
                user_age: np.reshape(self.users.values[user_id_val - 1][2], [1, 1]),
                user_job: np.reshape(self.users.values[user_id_val - 1][3], [1, 1]),
                movie_id: np.reshape(self.movies.values[self.movieid2idx[movie_id_val]][0], [1, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                dropout_keep_prob: 1}

            # Get Prediction
            inference_val = sess.run([inference], feed)
            return (inference_val)

    def create_movies_matrics(self):
        """
        生成电影特征矩阵
        :return:
        """
        loaded_graph = tf.Graph()  #
        movies_matrics = []
        with tf.Session(graph=loaded_graph) as sess:  #
            # Load saved model
            loader = tf.train.import_meta_graph(r'E:\dl_re_web\re_sys\recommend\model\model.meta')
            loader.restore(sess,  r'E:\dl_re_web\re_sys\recommend\model\model')
            # Get Tensors from loaded model
            uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = utils.get_tensors(loaded_graph)  # loaded_graph

            for item in self.movies.values:  #遍历每一个电影
                categories = np.zeros([1, 18])
                categories[0] = item.take(2)

                titles = np.zeros([1, self.sentences_size])
                titles[0] = item.take(1)

                feed = {
                    movie_id: np.reshape(item.take(0), [1, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    dropout_keep_prob: 1}

                movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
                movies_matrics.append(movie_combine_layer_flat_val)

        pickle.dump((np.array(movies_matrics).reshape(-1, 200)), open(r'E:\dl_re_web\re_sys\recommend\model\movies_matrics.pkl', 'wb'))
        print((np.array(movies_matrics).reshape(-1, 200).shape))
        # movie_matrics = pickle.load(open('./model/movie_matrics.p', mode='rb'))




    def create_users_matrics(self):
        """
        生成用户矩阵
        :return:
        """
        loaded_graph = tf.Graph()  #
        users_matrics = []

        with tf.Session(graph=loaded_graph) as sess:  #
            loader = tf.train.import_meta_graph(r'E:\dl_re_web\re_sys\recommend\model\model.meta')
            loader.restore(sess, r'E:\dl_re_web\re_sys\recommend\model\model')
            # Get Tensors from loaded model
            uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, \
            movie_combine_layer_flat, user_combine_layer_flat = utils.get_tensors(loaded_graph)  # loaded_graph
            for item in self.users.values:
                feed = {
                    uid: np.reshape(item.take(0), [1, 1]),
                    user_gender: np.reshape(item.take(1), [1, 1]),
                    user_age: np.reshape(item.take(2), [1, 1]),
                    user_job: np.reshape(item.take(3), [1, 1]),
                    dropout_keep_prob: 1}
                user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
                users_matrics.append(user_combine_layer_flat_val)

        pickle.dump((np.array(users_matrics).reshape(-1, 200)), open(r'E:\dl_re_web\re_sys\recommend\model\users_matrics.pkl', 'wb'))
        print((np.array(users_matrics).reshape(-1, 200).shape))
        # users_matrics = pickle.load(open('users_matrics.p', mode='rb'))


    def recommend_same_type_movie(self,movie_id_val, top_k=5):
        loaded_graph=tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            movies_matrics = self.movies_matrics  # (3883,200)
            probs_embeddings = (movies_matrics[self.movieid2idx[movie_id_val]]).reshape([1, 200])  # (1,200)

            norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movies_matrics), 1, keep_dims=True))#(3883,1)
            normalized_movie_matrics = movies_matrics / norm_movie_matrics #(3883,200)
            # 推荐同类型的电影

            probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))#(1,3883)
            sim = probs_similarity.eval()

            choice_movie = self.movies_orig[self.movieid2idx[movie_id_val]]
            print("您选择的电影是：{}".format(choice_movie))
            print("相似的电影是：")
            p = np.squeeze(sim)
            p[np.argsort(p)[:-top_k]] = 0
            p = p / np.sum(p)
            results = set()
            results_movies_names = []
            results_movies_ids = []
            while len(results) != 5:
                c = np.random.choice(3883, 1, p=p)[0]
                results.add(c)

            for val in (results):
                # print(val)
                results_movies_names.append(str(self.movies_orig[val][1]))
                results_movies_ids.append(str(self.movies_orig[val][0]))
                print(self.movies_orig[val])

            return choice_movie[1], results, results_movies_names,results_movies_ids

    def recommend_your_favorite_movie(self,user_id_val,top_k=5):
        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #
            movies_matrics = self.movies_matrics
            users_matrics = self.users_matrics
            # 推荐您喜欢的电影
            probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])
            probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movies_matrics))
            sim = (probs_similarity.eval())

        print("您的UserID："+str(user_id_val)+"--以下是给您推荐的电影：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            # print(val)
            print(self.movies_orig[val])

        return results

    def recommend_other_favorite_movie(self,movie_id_val, top_k = 5):
        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #

            movies_matrics = self.movies_matrics
            users_matrics=self.users_matrics

            print(movies_matrics.shape)
            probs_movie_embeddings = (movies_matrics[self.movieid2idx[movie_id_val]]).reshape([1, 200])
            probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
            favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]

            choice_movie=self.movies_orig[self.movieid2idx[movie_id_val]]
            print("您选择的电影是：{}".format(choice_movie))

            print("喜欢看这个电影的人是：{}".format(self.users_orig[favorite_user_id-1]))
            probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
            probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movies_matrics))
            sim = (probs_similarity.eval())

        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     print(sim.shape)
        #     print(np.argmax(sim, 1))
            p = np.argmax(sim, 1)
        print("喜欢看这个电影的人还喜欢看：")
        results = set()
        results_movie_names=[]
        results_movie_ids=[]
        for i in p:
            results.add(i)

        for val in (results):
            results_movie_names.append(str(self.movies_orig[val][1]))
            results_movie_ids.append(str(self.movies_orig[val][0]))
            print(self.movies_orig[val])
        return choice_movie[1],results,results_movie_names,results_movie_ids

    def recommend_by_movie(self,movie_id):
        choice_movie_1_name,list_same_movies,list_same_movies_names,list_same_movies_ids=self.recommend_same_type_movie(movie_id)
        choice_movie_2_name,list_pepole_like_movies,list_pepole_like_movies_names,list_pepole_like_movies_ids=self.recommend_other_favorite_movie(movie_id)

        return utils.format_names([choice_movie_1_name]),utils.format_names(list_same_movies_names),utils.format_names(list_pepole_like_movies_names),list_same_movies_ids,list_pepole_like_movies_ids


if __name__ == '__main__':
    model = Model(has_fit=True)
    # model.recommend_by_movie(1401)
    recommend_your_favorite_movie = model.recommend_your_favorite_movie(200)
    # model.fit()
    # model.create_users_matrics()
    # print(model.rating_movie(234,1401))
    # model.recommend_other_favorite_movie(140)
    # model.recommend_same_type_movie(1401)
    # model.recommend_same_type_movie(1400)
    # model.recommend_same_type_movie(1420)
    # choice_movie_name,list_same_movies_names,list_pepole_like_movies_names  =model.recommend_by_movie(1401)
    # print('----------')
    # print(choice_movie_name)
    # print(list_same_movies_names)
    # print(list_pepole_like_movies_names)

    # print("finish")
    # print(model.rating_movie_1(sess,loaded_graph,234,1401))
    # print(model.rating_movie_1(sess,loaded_graph,234,1400))
    #model.recommend_same_type_movie_1(1401)
    # model.recommend_same_type_movie(1401)
    model.recommend_your_favorite_movie(1401)
    model.recommend_same_type_movie(200)
    model.recommend_other_favorite_movie(1401)
