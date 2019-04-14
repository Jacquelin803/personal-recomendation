# -*-coding:utf8-*-
"""
author:jacquelin
date:2019
train gbdt model
"""
import xgboost as xgb
import sys
sys.path.append("../")
import util.get_feature_num as GF
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix


def get_train_data(train_file,feature_num_file):
    '''
    准备训练数据
    :param train_file:
    :param feature_num_file:
    :return:
    '''
    total_feature_num=GF.get_feature_num(feature_num_file)
    train_label=np.genfromtxt(train_file,dtype=np.int32,delimiter=',',usecols=-1)
    feature_list=range(total_feature_num)
    train_feature=np.genfromtxt(train_file,dtype=np.int32,delimiter=',',usecols=feature_list)
    return train_feature,train_label


def train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate):
    '''

    :param train_mat:
    :param tree_depth:
    :param tree_num:
    :param learning_rate:
    :return:
    '''
    para_dict={'max_depth':tree_depth,'eta':learning_rate,'objective':'reg:linear','silent':1}
    bst=xgb.train(para_dict,train_mat,tree_num)
    #print(xgb.cv(para_dict,train_mat,tree_num,nfold=5,metrics={'auc'}))
    return bst


def choose_parameter():
    '''
    :return: 不同参数的组合
    '''
    result_list=[]
    tree_depth_list=[4,5,6]
    tree_num_list=[10,50,100]
    learning_rate_list=[0.3,0.5,0.7]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth,ele_tree_num,ele_learning_rate))
    return result_list


def grid_search(train_mat):
    '''
    寻找最优超参数
    :param train_mat:
    :return:
    '''
    para_list=choose_parameter()
    for ele in para_list:
        (tree_depth,tree_num,learning_rate)=ele
        para_dict={'max_depth':tree_depth,'eta':learning_rate,'objective':'reg:linear','silent':1}
        result=xgb.cv(para_dict,train_mat,tree_num,nfold=5,metrics={'auc'})
        auc_score=result.loc[tree_num-1,['test-auc-mean']].values[0]     #在train_tree_model_core打印出的结果中最后一行是最高分，梯度上升，结果总会越来越好
        print('tree_depth:%s，tree_num:%s,learning_rate:%s,auc:%f' %(tree_depth,tree_num,learning_rate,auc_score))



def train_tree_model(train_file,feature_num_file,tree_model_file):
    '''

    :param train_file:
    :param feature_num_file:
    :param tree_model_file:
    :return:
    '''
    train_feature,train_label=get_train_data(train_file,feature_num_file)
    train_mat=xgb.DMatrix(train_feature,train_label)
    #grid_search(train_mat)      #此函数只要运行一次找到最有参数即可注释掉
    tree_num=100
    tree_depth=4
    learning_rate=0.3
    bst=train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate)
    bst.save_model(tree_model_file)


'''以上代码为了寻找最优超参数'''
'''以下代码训练树模型和logistics regression模型'''

def get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_depth):
    '''
    得到混合模型的特征数据，即数据转化
    :param tree_leaf:
    :param tree_num:
    :param tree_depth:
    :return:a sparse matricx 稀疏矩阵
    '''
    total_node_num=2**(tree_depth+1)-1
    leaf_num=2**tree_depth
    not_leaf_num=total_node_num-leaf_num
    total_col_num=leaf_num*tree_num    #所有列即所有特征是每棵树的叶子*所有树
    total_row_num=len(tree_leaf)     #原本转化来的tree_leaf，print(len(tree_leaf))可知每个数据样本都会转化为10棵树，进入不同的树落在不同的叶子节点上
    col=[]
    row=[]
    data=[]
    base_row_index=0
    for one_result in tree_leaf:
        base_col_index=0
        for fix_index in one_result:
            leaf_index=fix_index-not_leaf_num
            #可以借助这个例子理解，fix_index=15,leaf_index=15-15=0
            # 第一个样本落在10棵树上每棵的叶子节点[15 18 15 15 23 27 13 17 28 21]，深度为4，16个叶子节点，15个非叶子节点。
            # 第一位为15，表示落在第一棵树上第一个叶子节点上；最后一个21，表示落在第七个叶子节点上
            # 实际中特征：样本=1：100，深度为4时，总特征=2**4*10=160，总样本3W条
            leaf_index=leaf_index if leaf_index >=0 else 0
            col.append(base_col_index+leaf_index)
            row.append(base_row_index)
            data.append(1)
        base_row_index+=1    #每算完一条one_result也即一个样本都要在行上加1
    total_feature_list=coo_matrix((data,(row,col)),shape=(total_row_num,total_col_num))
    return total_feature_list

def get_mix_model_tree_info():
    tree_num=10
    tree_depth=4
    learning_rate=0.3
    result=(tree_depth,tree_num,learning_rate)
    return result


def train_tree_and_lr_model(train_file,feature_num_file,mix_tree_model_file,mix_lr_model_file):
    '''
    树模型与lr是分开训练的，分别保存两个训练文件
    混合模型原理https://zhuanlan.zhihu.com/p/42123341
    :param train_file:
    :param feature_num_file:
    :param mix_tree_model_file: XGboost树模型文件
    :param mix_lr_model_file: logistics模型
    '''
    #树模型的获取
    train_feature,train_label=get_train_data(train_file,feature_num_file)
    train_mat=xgb.DMatrix(train_feature,train_label)
    (tree_depth,tree_num,learning_rate)=get_mix_model_tree_info()
    print(tree_depth,tree_num,learning_rate)
    bst=train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate)
    bst.save_model(mix_tree_model_file)

    #logistic regression模型的获取：用树结构处理数据形成特征后fit logistic模型
    tree_leaf=bst.predict(train_mat,pred_leaf=True)      #样本落在哪个节点上
    print(len(tree_leaf))
    print (tree_leaf[0])
    #第一个样本落在10棵树上每棵的叶子节点[15 18 15 15 23 27 13 17 28 21]，深度为4，16个叶子节点，15个非叶子节点。
    # 第一位为15，表示落在第一棵树上第一个叶子节点上；最后一个21，表示落在第七个叶子节点上
    #实际中特征：样本=1：100，深度为4时，总特征=2**4*10=160，总样本3W条
    #sys.exit()
    total_feature_list=get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_depth)
    lr_clf=LRCV(Cs=[1.0],penalty='l2',dual=False,tol=0.0001,max_iter=500,cv=5)
    lr_clf=lr_clf.fit(total_feature_list,train_label)
    scores = list (lr_clf.scores_.values ())[0]
    print ("diffC:%s" % (','.join([str(ele) for ele in scores.mean(axis=0)])))
    print ("Accuracy:%f(+-%0.2f)" % (scores.mean(), scores.std() * 2))
    lr_clf = LRCV(Cs=[1.0], penalty='l2', dual=False, tol=0.0001, max_iter=500, scoring='roc_auc', cv=5).fit(
        total_feature_list, train_label)
    scores = list (lr_clf.scores_.values ())[0]
    print ("diffC:%s" % (','.join([str(ele) for ele in scores.mean(axis=0)])))
    print ("AUC:%f,(+-%0.2f)" % (scores.mean(), scores.std() * 2))
    fw = open(mix_lr_model_file, "w+")
    coef = lr_clf.coef_[0]
    fw.write(','.join([str(ele) for ele in coef]))



if __name__=='__main__':
    #train_tree_model('../data/train_file','../data/feature_num_file','../data/xgb.model')
    #train_tree_and_lr_model('../data/train_file','../data/feature_num_file','../data/xgb_mix_model','../data/lr_coef_mix_model')
    if len(sys.argv)==4:
        train_file=sys.argv[1]
        feature_num_file=sys.argv[2]
        tree_model=sys.argv[3]
        train_tree_model(train_file,feature_num_file,tree_model)
    elif len(sys.argv)==5:
        train_file=sys.argv[1]
        feature_num_file=sys.argv[2]
        tree_mix_model=sys.argv[3]
        lr_coef_mix_model=sys.argv[4]
        train_tree_and_lr_model(train_file,feature_num_file,tree_mix_model,lr_coef_mix_model)
    else:
        print('train gbdt model usage: python xx.py train_file feature_num_file tree_model')
        print ("train lr_gbdt model usage: python xx.py train_file feature_num_file tree_mix_model lr_coef_mix_model")
        sys.exit()
