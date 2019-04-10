# -*-coding:utf8-*-
"""
author:jacquelin
date:2019
feature selection and data selection
"""
import pandas as pd
import numpy as np
import operator
import sys


def get_input(input_train_file,input_test_file):
    '''
    选取特征和特征数据格式转换
    :param input_train_file:
    :param input_test_file:
    :return: 2 dataframe
    '''
    dtype_dict={'age':np.int32,
                'education-num':np.int32,
                'capital - gain':np.int32,
                'capital - loss':np.int32,
                'hours - per - week':np.int32}
    use_list=list(range(15))
    use_list.remove(2)
    train_data_df=pd.read_csv(input_train_file,sep=',',header=0,dtype=dtype_dict,na_values='?',usecols=use_list)
    #print(train_data_df.shape)
    train_data_df=train_data_df.dropna(axis=0,how='any')
    #print (train_data_df.shape)
    test_data_df=pd.read_csv(input_test_file,sep=',',header=0,dtype=dtype_dict,na_values='?',usecols=use_list)
    test_data_df=train_data_df.dropna(axis=0,how='any')
    return train_data_df,test_data_df

#转换思想：先将各种分散特征的值提取，将文字用函数转换为数字；再用函数对整个dataframe进行处理
#对y值得转化

def label_trans(x):
    '''
    y值转换
    :param x:
    :return:
    '''
    if x=='<=50K':
        return '0'
    if x=='>50K':
        return '1'
    return '0'


def process_label_feature(label_feature_str,df_in):
    '''
    对整个dataframe的y值也即label进行转换
    :param label_feature_str:
    :param df_in:
    :return:
    '''
    df_in.loc[:,label_feature_str]=df_in.loc[:,label_feature_str].apply(label_trans)

#对各特征的转换
def dict_trans(dict_in):
    '''
    将文字特征转化为数字
    :param dict_in: original_dict
    :return: a  dict key str,value int 0,1,2
    '''
    output_dict={}
    index=0
    for combination in sorted(dict_in.items(),key=operator.itemgetter(1),reverse=True):
        output_dict[combination[0]]=index
        index+=1
    return output_dict


def dis_to_feature(x,feature_dict):
    '''

    :param x:
    :param feature_dict:
    :return:
    '''
    output_list=[0]*len(feature_dict)
    if x not in feature_dict:
        return ','.join([str(ele) for ele in output_list])
    else:
        index=feature_dict[x]   #feature_dict里x元素对应的值
        output_list[index]=1    #值就是向量里1的位置对应{'Private': 0, 'Self-emp-not-inc': 1, 'Local-gov': 2, 'State-gov': 3, 'Self-emp-inc': 4, 'Federal-gov': 5, 'Without-pay': 6}
    return ','.join([str(ele) for ele in output_list])



def process_dis_feature(feature_str,df_train,df_test):
    '''
    处理离散数据process dis feature for lr train model
    :param feature_str:
    :param df_train:
    :param df_test:
    :return: the dim of the feature output

    '''
    original_dict=df_train.loc[:,feature_str].value_counts().to_dict()
    feature_dict=dict_trans(original_dict)
    df_train.loc[:, feature_str]=df_train.loc[:, feature_str].apply(dis_to_feature,args=(feature_dict, ))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply (dis_to_feature , args=(feature_dict,))
    #print(df_train.loc[:3,feature_str])
    #print(feature_dict)
    return len(feature_dict)


def list_trans(input_dict):
    '''
    对连续型变量描述后形成字典格式文件，要将字典转化为list
    :param input_dict: {'count': 30162.0, 'mean': 38.437901995888865, 'std': 13.134664776856338, 'min': 17.0,
                            '25%': 28.0, '50%': 37.0, '75%': 47.0, 'max': 90.0}
    :return:
    '''
    output_list=[0]*5
    key_list=['min','25%','50%','75%','max']
    for index in range(len(key_list)):
        fix_key=key_list[index]
        if fix_key not in input_dict:
            print('error')     #新输入的字典里没有本脚本要求的元素，说明新输入的东西有误
            sys.exit()
        else:
            output_list[index]=input_dict[fix_key]
    return output_list


def con_to_feature(x,feature_list):
    '''
    原数据中数值应该处在哪个区间，进行定义
    :param x:element
    :param feature_list: list for feature trans
    :return:
    '''
    feature_len=len(feature_list)-1
    result=[0]*feature_len
    for index in range(feature_len):
        if x>=feature_list[index] and x<=feature_list[index+1]:
            result[index]=1
            return ','.join([str(ele) for ele in result])
    return ','.join([str(ele) for ele in result])



def process_con_feature(feature_str,df_train,df_test):
    '''
    处理连续性特征的函数：将连续型特征的值标准化
    :param feature_str:
    :param df_train:
    :param df_test:
    :return:
    '''
    original_dict=df_train.loc[:,feature_str].describe().to_dict()
    #测试用print(original_dict)
    feature_list=list_trans(original_dict)
    df_train.loc[:,feature_str]=df_train.loc[:,feature_str].apply(con_to_feature,args=(feature_list,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply (con_to_feature, args=(feature_list,))
    #print(feature_list)
    #print(df_train.loc[:3,feature_str])
    return len (feature_list)-1

def output_file(df_in,out_file):
    fw=open(out_file,'w+')
    for row_index in df_in.index:
        outline=','.join([str(ele) for ele in df_in.loc[row_index].values])
        fw.write(outline+'\n')
    fw.close()




def add(str_one,str_two):
    '''
    str_one:ABCD
    str_two:EFGH
    16种组合情况：AE，AF。。。DH
    AE:0*4+1=1,新list中1在0的位置，第一个
    AF:0*4+2=2,第二个
    CF:2*4+2=10,第十个是1
0    0,0,1,0
1    0,0,0,1
2    0,0,1,0
3    0,0,0,1
4    1,0,0,0
Name: age, dtype: object
0    0,0,0,1
1    1,0,0,0
2    1,0,0,0
3    1,0,0,0
4    1,0,0,0
Name: capital-gain, dtype: object
0    0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0
1    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0
2    0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0
3    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0
4    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    :param str_one:0,0,1,0
    :param str_two:0,1,0,0
    :return:
    '''
    list_one=str_one.split(',')
    list_two = str_two.split (',')
    list_one_len=len(list_one)
    list_two_len = len (list_two)
    return_list=[0]*(list_one_len*list_two_len)
    try:
        index_one=list_one.index('1')
    except:
        index_one=0
    try:
        index_two=list_two.index('1')
    except:
        index_two=0
    return_list[index_one*list_two_len+index_two]=1

    return ','.join([str(ele) for ele in return_list])



def combine_feature(feature_one,feature_two,new_feature,train_data_df,test_data_df,feature_num_dict):
    '''

    :param feature_one:
    :param feature_two:
    :param new_feature:
    :param train_data_df:
    :param test_data_df:
    :param feature_num_dict:
    :return:
    '''
    train_data_df[new_feature]=train_data_df.apply(lambda row:add(row[feature_one],row[feature_two]),axis=1)
    test_data_df[new_feature] = test_data_df.apply (lambda row: add (row[feature_one], row[feature_two]), axis=1)
    if feature_one not in feature_num_dict:
        print('error')
    if feature_two not in feature_num_dict:
        print('error')
        sys.exit()
    return feature_num_dict[feature_one]*feature_num_dict[feature_two]



def ana_train_data(input_train_data,input_test_data,out_train_file,out_test_file,feature_num_file):

    
    '''
    :param input_train_data:
    :param input_test_data:
    :param output_train_file:
    :param output_test_file:
    '''
    train_data_df,test_data_df=get_input(input_train_data,input_test_data)

    label_feature_str='label'
    dis_feature_list=["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"]
    con_feature_list=["age","education-num","capital-gain","capital-loss","hours-per-week"]
    index_list = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    process_label_feature(label_feature_str,train_data_df)
    process_label_feature (label_feature_str, test_data_df)

    dis_feature_num= 0
    con_feature_num = 0
    feature_num_dict={}

    #对每个离散特征转换
    for dis_feature in dis_feature_list:
        tmp_feature_num=process_dis_feature(dis_feature,train_data_df,test_data_df)
        dis_feature_num+=tmp_feature_num
        feature_num_dict[dis_feature]=tmp_feature_num

    #对每个连续特征转换
    for con_feature in con_feature_list:
        tmp_feature_num=process_con_feature(con_feature,train_data_df,test_data_df)
        con_feature_num+=tmp_feature_num
        feature_num_dict[con_feature] = tmp_feature_num
    new_feature_len=combine_feature("age", "capital-gain", "age_gain", train_data_df, test_data_df, feature_num_dict)
    #print(new_feature_len)
    #print(train_data_df['age'][:8])
    #print (train_data_df['capital-gain'][:8])
    #print (train_data_df['age_gain'][:8])
    new_feature_len_two = combine_feature ("capital-gain", "capital-loss", "loss_gain", train_data_df, test_data_df,
                                           feature_num_dict)
    print(new_feature_len_two)
    train_data_df = train_data_df.reindex (columns=index_list + ["age_gain", "loss_gain", "label"])
    test_data_df = test_data_df.reindex (columns=index_list + ["age_gain", "loss_gain", "label"])


    #print(dis_feature_num)
    #print (con_feature_num)
    output_file(train_data_df,out_train_file)
    output_file(test_data_df,out_test_file)
    fw = open (feature_num_file, "w+")
    fw.write ("feature_num=" + str (dis_feature_num + con_feature_num + new_feature_len + new_feature_len_two))


if __name__=='__main__':
    if len(sys.argv) <6:     #在run_total.sh运行第一个py文件有6个参数的
        print('usage: python xx.py origin_train origin_test train_file test_file feature_num_file')
        sys.exit()
    else:
        origin_train=sys.argv[1]
        origin_test=sys.argv[2]
        train_file=sys.argv[3]
        test_file=sys.argv[4]
        feature_num_file=sys.argv[5]
        ana_train_data(origin_train,origin_test,train_file,test_file,feature_num_file)


    #ana_train_data('../data/train.txt','../data/test.txt','../data/train_file','../data/test_file','../data/feature_num_file')








