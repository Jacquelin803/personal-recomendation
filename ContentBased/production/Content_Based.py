#-*-coding:utf8-*-
"""
author:jacquelin
date:2019
get up and online recommendation
"""

from __future__ import division
import os
import operator
import  sys
sys.path.append("../")
import util.read as read

def get_up(item_cate,input_file):
    '''

    :param item_cate: {itemid: {cate:ratio}}
    :param input_file: ratings.txt用户行为文件
    :return: {userid:[(cate1,ratio1),(ctae2,ratio2)]}
    '''
    if not os.path.exists(input_file):
        return {}

    linenum=0
    score_thr=4.0
    record={}
    up={}
    topK=2
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<4:
            continue
        userid,itemid,rating,timestamp=item[0],item[1],float(item[2]),int(item[3])
        if rating<score_thr:
            continue     #不要小于4分的评分数据
        if itemid not in item_cate:
            continue
        time_score=get_time_score(timestamp)
        if userid  not in record:
            record[userid]={}
        for fix_cate in item_cate[itemid]:
            if fix_cate not in record[userid]:
                record[userid][fix_cate]=0    #record意在记录userID行为过的类别，类别值有其得分{itemid:{cate:score}}
            record[userid][fix_cate]+=rating*time_score*item_cate[itemid][fix_cate]   #record虽然与item_cate格式相同，但score是综合计算后的值而非ratio
    fp.close()
    for userid in record:
        if userid not in up:
            up[userid]=[]
        total_score=0
        for combination in sorted(record[userid].items(),key=operator.itemgetter(1),reverse=True)[:topK]:
            up[userid].append((combination[0],combination[1]))
            total_score+=combination[1]
            #print(total_score)
            #对score归一化
        for index in range(len(up[userid])):
            up[userid][index]=(up[userid][index][0],round(up[userid][index][1]/total_score,3))
                #上式计算的是：在up里第index个userid的cate对应的标准化后的分数,可以看出哪个类别多一些
    return up

def get_time_score(timestamp):
    '''
    时间近的得分就高
    :param timestamp:
    :return:
    '''
    fix_timestamp=1476086345
    total_sec=60*60*24  #结果计算成天
    delta=(fix_timestamp-timestamp)/total_sec/100

    return round(1/(1+delta),3)

def recom(cate_item_sort,up,userid,topk=10):
    '''
    根据给的userid推荐产品给他
    :param cate_item_sort:
    :param up:
    :param userid:
    :param topK:
    :return:
    '''
    if userid not in up:
        return {}
    recom_result={}
    if userid not in recom_result:
        recom_result[userid]=[]
    for combination in up[userid]:
        cate=combination[0]
        ratio=combination[1]
        #print(ratio) 各种类别应该推荐的比例
        print(combination)
        num=int(topk*ratio)+1
        if cate not in cate_item_sort:
            continue
        recom_list=cate_item_sort[cate][:num]
        recom_result[userid]+=recom_list
    return recom_result


def run_main():
    ave_score=read.get_ave_score('../data/ratings.txt')
    item_cate,cate_item_sort=read.get_item_cate(ave_score,'../data/movies.txt')
    up=get_up(item_cate,'../data/ratings.txt')
    #print(up['1'])
    print(recom(cate_item_sort,up,'1'))

if __name__=='__main__':
    run_main()



