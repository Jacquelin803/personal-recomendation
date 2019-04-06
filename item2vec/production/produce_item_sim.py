'''
生成item_vec文件流程：
（Terminal里运行）
1.编写train.sh程序
train_file=$1
../bin/word2vec -train $train_file -output ../data/item_vec.txt -size 128 -window 5 -sample 1e-3 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 50
binary一般设为0，设为1会生成二进制文件
2.ls
cd production/
ls
sh train.sh ../data/train_data.txt
3.等待或者敲使用item_vec的py程序。binary为0且迭代次数iter=50用了5分钟左右

'''

'''
produce item sim file生成相似度矩阵
'''
import os
import numpy as np
import operator
import sys

def load_item_vec(input_file):
    '''
    :param input_file: item_vec file
    :return:
        dict  key:itemid value:np.array([num1,num2...)
    '''
    if not os.path.exists(input_file):
        return {}
    linenum=0
    item_vec={}
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split()    #按照空格分割
        if len(item)<129:
            continue
        itemid=item[0]
        if itemid=='</s>':
            continue
        item_vec[itemid]=np.array([float(ele) for ele in item[1:]])
    fp.close()
    return item_vec

def cal_item_sim(item_vec,itemid,output_file):
    '''
    给出推荐产品id及其相似度
    :param item_vec: item embedding vector
    :param itemid: 要根据这个产品推荐其他产品
    :param output_file: recommend result
    '''
    if itemid not in item_vec:
        return
    score={}
    topK=10
    fix_item_vec=item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid==itemid:
            continue        #排除自身
        tmp_itemvec=item_vec[tmp_itemid]  #其他产品的矩阵
        denominator=np.linalg.norm(fix_item_vec)*np.linalg.norm(tmp_itemvec)
        if denominator==0:
            score[tmp_itemid]=0   #当分母等于0，让这两个向量的距离cos值为0
        else:
            score[tmp_itemid]=round(np.dot(fix_item_vec,tmp_itemvec)/denominator,3)
    fw=open(output_file,'w+')
    out_str=itemid+'\t'
    tmp_list=[]
    for combination in sorted(score.items(),key=operator.itemgetter(1),reverse=True)[:topK]:
        tmp_list.append(combination[0]+'_'+str(combination[1]))
    out_str+=';'.join(tmp_list)
    fw.write(out_str+'\n')
    fw.close()

def run_main(input_file,output_file):
    '''
    将整个流程运行
    :param input_file:
    :param output_file:
    :return:
    '''
    item_vec=load_item_vec(input_file)
    cal_item_sim(item_vec,'27',output_file)




if __name__=='__main__':
    if len(sys.argv)<3:
        print('usage: python **.py inputfile outputfile')
        sys.exit()
    else:
        inputfile=sys.argv[1]
        outputfile=sys.argv[2]
        run_main(inputfile,outputfile)




