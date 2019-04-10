# -*-coding:utf8-*-
"""
author:jacquelin
date:2019
use lr model to check the performance in test file
"""
from __future__ import division
import numpy as np
from sklearn.externals import joblib
import math
import sys
sys.path.append("../")
import util.get_feature_num as GF


def get_test_data(test_file,feature_num_file):
    '''
    将测试数据和特征个数解析
    :param test_file:
    :param feature_num_file:
    :return: 2 np.array:  test_feature,test_label
    '''
    total_feature_num=GF.get_feature_num(feature_num_file)
    test_label=np.genfromtxt(test_file,dtype=np.float32,delimiter=',',usecols=-1)
    feature_list=range(total_feature_num)
    test_feature=np.genfromtxt(test_file,dtype=np.float32,delimiter=',',usecols=feature_list)
    return test_feature,test_label


def predict_by_lr_model(test_feature,lr_model):
    '''
    计算通过model预测的值
    :param test_feature:
    :param le_model:
    :return:
    '''
    result_list=[]
    prob_list=lr_model.predict_proba(test_feature)   #模型自带predict_proba函数
    for index in range(len(prob_list)):
        result_list.append(prob_list[index][1])    #取出预测大于阈值的概率，也即第二个数，index=1
    return result_list




def predict_by_lr_coef(test_feature,lr_coef):
    '''
    通过系数计算的预测值
    :param test_feature:
    :param lr_coef:
    :return:
    '''
    sigmoid_func=np.frompyfunc(sigmoid,1,1)
    return sigmoid_func(np.dot(test_feature,lr_coef))


def sigmoid(x):
    '''
    predict_by_lr_coef函数用到的
    :param x:
    :return:
    '''
    return 1/(1+math.exp(-x))


def get_auc(predict_list, test_label):
    '''
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    auc公式解析https://blog.csdn.net/Jacquelin_1/article/details/89142037
    :param preidct_list:
    :param test_label:
    :return:
    '''
    total_list=[]
    for index in range(len(predict_list)):
        predict_score=predict_list[index]
        label=test_label[index]
        total_list.append((label,predict_score))   #将预测值提取
    sorted_total_list=sorted(total_list,key=lambda ele:ele[1])
    neg_num=0
    pos_num=0
    count=1   #rank要排序
    total_pos_index=0
    for combination in sorted_total_list:
        label,predict_score=combination
        if label==0:
            neg_num+=1     #负样本
        else:
            pos_num+=1      #正样本
            total_pos_index+=count
        count+=1    #rank排序是对所有正负样本排序的，每过index-(pos_)
    auc_score=(total_pos_index-(pos_num)*(pos_num+1)/2)/(pos_num*neg_num)
    print('auc:%.5f' %(auc_score))


def get_accuracy(predict_list,test_label):
    '''
    预测的准确率
    :param predict_list:
    :param test_label:
    :return:
    '''
    score_thr=0.5    #阈值
    right_num=0
    for index in range(len(predict_list)):
        predict_score=predict_list[index]
        if predict_score>=score_thr:
            predict_label=1
        else:
            predict_label=0
        if predict_label==test_label[index]:
            right_num+=1
    total_num=len(predict_list)
    accuracy_score=right_num/total_num
    print('accuracy:%.5f'%(accuracy_score))



def run_check_core(test_feature,test_label,model,score_func):
    '''
    需要根据两种结果进行测试
    :param test_feature:
    :param test_label:
    :param model: lr_coef,lr_model
    :param score_func:predict_by_lr_model/predict_by_lr_coef
    '''
    predict_list=score_func(test_feature,model)   #得到预测值
    get_auc (predict_list, test_label)
    get_accuracy (predict_list, test_label)

def run_check(test_file,lr_coef_file,lr_model_file,feature_num_file):
    '''

    :param test_file:
    :param lr_coef_file:
    :param lr_model_file:
    :param feature_num_file:
    :return:
    '''

    test_feature,test_label=get_test_data(test_file,feature_num_file)
    #lr系数
    lr_coef=np.genfromtxt(lr_coef_file,dtype=np.float32,delimiter=',')
    #lr模型
    lr_model=joblib.load(lr_model_file)

    run_check_core(test_feature,test_label,lr_model,predict_by_lr_model)
    run_check_core (test_feature, test_label, lr_coef, predict_by_lr_coef)
    '''
    predict_by_lr_model/predict_by_lr_coef得到auc是一样的
    auc: 0.89701
    auc: 0.89701
    '''

if __name__=='__main__':
    if len(sys.argv) <5:
        print('"usage: python xx.py test_file coef_file model_file feature_num_file')
        sys.exit()
    else:
        test_file=sys.argv[1]
        lr_coef_file=sys.argv[2]
        lr_model_file=sys.argv[3]
        feature_num_file=sys.argv[4]
        run_check(test_file,lr_coef_file,lr_model_file,feature_num_file)

    #run_check('../data/test_file','../data/lr_coef','../data/lr_model_file','../data/feature_num_file')



'''
run_total.sh主流程文件运行步骤：
(base) jacquelindeMacBook-Pro:LR jacquelin$ ls
data            production      util
(base) jacquelindeMacBook-Pro:LR jacquelin$ cd production/
(base) jacquelindeMacBook-Pro:production jacquelin$ ls
__init__.py             ana_train_data.py       run_total.sh            train.py
ana_2.py                check.py                tes.py
(base) jacquelindeMacBook-Pro:production jacquelin$ sh run_total.sh
'''







