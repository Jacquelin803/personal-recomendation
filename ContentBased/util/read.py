#-*-coding:utf8-*-
"""
author:jacquelin
date:2019
some util function
"""

from __future__ import division
import os
import operator


def get_ave_score(input_file):
    '''
    item ave score from ratings
    :param input_file: ratings
    :return: a dict , key:itemid value:ave_score
    '''
    if not os.path.exists(input_file):
        return {}
    linenum=0
    record={}
    ave_score={}
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<4:
            continue
        userid,itemid,ratings=item[0],item[1],float(item[2])
        if itemid not in record:
            record[itemid]=[0,0] #评分加和，评分次数，为求均分做数据准备
        record[itemid][0]+=ratings
        record[itemid][1]+=1
    fp.close()
    for itemid in record:
            ave_score[itemid]=round(record[itemid][0]/record[itemid][1],3)
    return ave_score

def get_item_cate(ave_socre,input_file):
    '''
    提取item类别及该类别下的item为推荐做准备
    :param ave_socre: 上方函数得出的均分
    :param input_file: movies产品文件
    :return:
        a dict {itemid:{cate:ratio}}    item_cate
        a dict {cate:[itemid1,itemid2,itemid3]}    cate_item_sort
    '''
    if not os.path.exists(input_file):
        return {},{}
    linenum=0
    item_cate={}
    record={}
    cate_item_sort={}
    topK=100
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<3:
            continue
        itemid=item[0]
        cate_str=item[-1]
        cate_list=cate_str.strip().split('|')
        ratio=round(1/len(cate_list),3)
        if itemid not in item_cate:
            item_cate[itemid]={}
        for fix_cate in cate_list:
            item_cate[itemid][fix_cate]=ratio
    fp.close()
    #将均分挂在itemID上
    for itemid in item_cate:
        for cate in item_cate[itemid]:
            if cate not in record:
                record[cate]={}
            itemid_rating_score=ave_socre.get(itemid,0)      #get函数返回元素对应的值
            record[cate][itemid]=itemid_rating_score
    for cate in record:
        if cate not in cate_item_sort:
            cate_item_sort[cate]=[]
        for combination in sorted(record[cate].items(),key=operator.itemgetter(1),reverse=True)[:topK]:
            cate_item_sort[cate].append(combination[0])
    return item_cate,cate_item_sort

def get_latest_timestamp(input_file):
    if not os.path.exists(input_file):
        return
    linenum=0
    latest=0
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<4:
            continue
        timestamp=int(item[3])
        if timestamp>latest:
            latest=timestamp
    fp.close()
    print(latest)

if  __name__=='__main__':
    ave_score=get_ave_score('../data/ratings.txt')
    #print(len(ave_score))
    #print(ave_score['31'])
    item_cate,cate_item_sort=get_item_cate(ave_score,'../data/movies.txt')
    print(item_cate['1'])
    print(cate_item_sort['Children'])
    #get_latest_timestamp('../data/ratings.txt')



















