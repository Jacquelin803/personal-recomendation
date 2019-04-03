

import numpy as np
import sys
sys.path.append('../util')
import util.read as read
import operator


def lfm_train(train_data,F,alpha,beta,step):
    '''

    :param train_data: read模块读取的结果
    :param F: 隐变量（隐特征）
    :param alpha:正则化参数
    :param beta:学习率
    :param step:迭代次数
    :return:
        dict: key itemid, value:np.ndarray
        dict: key userid, value:np.ndarray
    '''
    user_vec={}
    item_vec={}

    for step_index in range(step):
        for data_distance in train_data:
            userid,itemid,label=data_distance
            if userid not in user_vec:
                user_vec[userid]=init_model(F)
            if itemid not in item_vec:
                item_vec[itemid]=init_model(F)
            delta=label-model_predict(user_vec[userid],item_vec[itemid])
            for index in range(F):
                user_vec[userid][index]+=beta*(delta*item_vec[itemid][index]-alpha*user_vec[userid][index])
                item_vec[itemid][index]+= beta * (delta * user_vec[userid][index] - alpha * item_vec[itemid][index])
        beta=beta*0.9   #梯度下降
    return user_vec,item_vec

def init_model(vector_len):
    '''
    此函数意在给user_vec[userid]赋一个合理的值
    :param vector_len: 特征数
    :return: a ndarray
    '''
    return np.random.rand(vector_len)

def model_predict(user_vector,item_vector):
    '''
    两个向量间的余弦
    :param user_vec:
    :param item_vec:
    :return: a num
    '''
    res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res
def model_train_process():
    train_data=read.get_train_data('../data/ratings.txt')
    user_vec,item_vec=lfm_train(train_data,50,0.01,0.1,50)
    for userid in user_vec:
        recom_result=give_recom_result(user_vec,item_vec,userid)
        ana_recom_result(train_data,userid,recom_result)

def give_recom_result(user_vec,item_vec,userid):
    '''
    对于每一位用户userID都给出推荐结果
    :param user_vec:
    :param item_vec:
    :param userid:
    :return:  a list:[(itemid1,score1),(itemid2,score2)]
    '''
    fix_num=10   #给每位用户推荐10条
    if userid not in user_vec:
        return []      #新用户没有原始数据用作分析无法给出推荐
    record= {}
    recom_list=[]
    user_vector=user_vec[userid]
    for itemid in item_vec:
        item_vector=item_vec[itemid]
        res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))    #计算item与用户间的距离
        record[itemid]=res
    for combination in sorted(record.items(),key=operator.itemgetter(1),reverse=True)[:fix_num]:
        itemid=combination[0]
        score=round(combination[1],3)
        recom_list.append((itemid,score))
    return recom_list

def ana_recom_result(train_data,userid,recom_list):
    '''
    对算法推荐给用户的item进行分析
    :param train_data:
    :param userid:
    :param recom_list:
    :return:
    '''
    item_info=read.get_item_info('../data/movies.txt')
    for data_distance in train_data:
        tmp_userid,itemid,label=data_distance
        if tmp_userid==userid and label==1:
            print('user like:',item_info[itemid])
    print('recom result')
    for combination in recom_list:
        print(item_info[combination[0]])




if __name__ == "__main__":
    model_train_process()



