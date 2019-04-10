# -*-coding:utf8-*-
"""
author:jacquelin
date:2019
train lr model
"""
import sys
sys.path.append("../")
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.externals import joblib   #将模型整体实例化出来
import util.get_feature_num as GF
import numpy as np


def train_lr_model(train_file,model_coef,model_file,feature_num_file):


    total_feature_num=GF.get_feature_num(feature_num_file)

    train_label = np.genfromtxt (train_file, dtype=np.int32, delimiter=",", usecols=-1)
    feature_list = range (total_feature_num)
    train_feature = np.genfromtxt (train_file, dtype=np.int32, delimiter=",", usecols=feature_list)

    lr_cf=LRCV(Cs=[1],penalty='l2',tol=0.0001,max_iter=500,cv=5).fit(train_feature,train_label)
    scores=list(lr_cf.scores_.values())[0]
    print('diff:%s' %(','.join([str(ele) for ele in scores.mean(axis=0)])))
    print('Accuracy:%s (+-%0.2f)' %(scores.mean(),scores.std()*2))
    #平均值0.842616805029923上下0.01就可覆盖90%的值，说明0.842616805029923是很靠谱的

    lr_cf = LRCV (Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5,scoring='roc_auc').fit (train_feature, train_label)
    scores = list (lr_cf.scores_.values ())[0]
    print ('diff:%s' % (','.join ([str (ele) for ele in scores.mean (axis=0)])))
    print ('AUC:%s (+-%0.2f)' % (scores.mean (), scores.std () * 2))

    coef=lr_cf.coef_[0]
    fw=open(model_coef,'w+')
    fw.write(','.join(str(ele) for ele in coef))
    fw.close()
    joblib.dump(lr_cf,model_file)


if __name__=='__main__':
    if len(sys.argv) <5:
        print('usage: python xx.py train_file lr_coef_file lr_model_file featuren_num_file')
        sys.exit()
    else:
        train_file=sys.argv[1]
        lr_coef_file=sys.argv[2]
        lr_model_file=sys.argv[3]
        feature_num_file=sys.argv[4]
        train_lr_model(train_file,lr_coef_file,lr_model_file,feature_num_file)




    #train_lr_model('../data/train_file','../data/lr_coef','../data/lr_model_file','../data/feature_num_file')










