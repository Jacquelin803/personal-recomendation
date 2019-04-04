
'''
公示推导：

初始赋予 PR(A)=1,PR(B)=PR(C)=PR(a)=PR(b)=PR(c)=PR(d)=0，即对于A来说，他自身的重要度为满分，其他节点的重要度均为0。

然后开始在图上游走。每次都是从PR不为0的节点开始游走，往前走一步。继续游走的概率是α，停留在当前节点的概率是1−α。

第一次游走， 从A节点以各自50%的概率走到了a和c，这样a和c就分得了A的部分重要度，PR(a)=PR(c)=α∗PR(A)∗0.5。最后PR(A)变为1−α。第一次游走结束后PR不为0的节点有A a c。

第二次游走，分别从节点A a c开始，往前走一步。这样节点a分得A 12∗α的重要度，节点c分得A 12∗α的重要度，节点A分得a 12∗α的重要度，节点A分得c 13∗α的重要度，节点B分得a 12∗α的重要度，节点B分得c 13∗α的重要度，节点C分得c 13∗α的重要度。最后PR(A)要加上 1−α

见大神博文，讲的很清楚，https://www.cnblogs.com/zhangchaoyang/articles/5470763.html
'''

from __future__ import division
import sys
sys.path.append("../util")
import util.read as read
import operator
import util.matrix_util as mat_util
from scipy.sparse.linalg import gmres
import numpy as np
'''
def personal_rank(graph,root,alpha,iter_num,recom_num=10):
    
    personal_rank算法公式底层实现
    :param graph: read里得到的二分图
    :param root: 要给推荐产品的用户
    :param alpha: 在这个节点继续随机游走的概率
    :param iter_num: 迭代次数
    :param recom_num: 推荐个数
    :return:
        a dict:key-itemid,value-pr
    
    rank ={}
    rank ={point:0 for point in graph}
    rank[root]=1
    recom_result={}
    for iter_index in range(iter_num):
        tmp_rank={}
        tmp_rank={point:0 for point in graph}
        for out_point,out_dict in graph.items():
            for inner_point,value in graph[out_point].items():
                tmp_rank[inner_point]+=round(alpha*rank[out_point]/len(out_dict),4)
                if inner_point==root:       #若是恰好是自己要返回去，就要加上停下来的概率
                    tmp_rank[inner_point]+=round(1-alpha,4)
        if tmp_rank==rank:
            #print('out'+str(iter_index))
            break
        rank=tmp_rank
    right_num=0

    for combination in sorted(rank.items(),key=operator.itemgetter(1),reverse=True):
        point,pr_score=combination[0],combination[1]
        if len(point.split('-'))<2:
            continue
        if point in graph[root]:
            continue
        recom_result[point]=round(pr_score,4)
        right_num+=1
        if right_num>recom_num:
            break
    return recom_result
'''

def personal_rank(graph, root, alpha, iter_num, recom_num= 10):
    """
    Args
        graph: user item graph
        root: the  fixed user for which to recom
        alpha: the prob to go to random walk
        iter_num:iteration num
        recom_num: recom item num
    Return:
        a dict, key itemid, value pr
    """

    rank = {}
    rank = {point:0 for point in graph}
    rank[root] = 1
    recom_result = {}
    for iter_index in range(iter_num):
        tmp_rank = {}
        tmp_rank = {point:0 for point in graph}
        for out_point, out_dict in graph.items():
            for inner_point, value in graph[out_point].items():
                tmp_rank[inner_point] += round(alpha*rank[out_point]/len(out_dict), 4)
                if inner_point == root:
                    tmp_rank[inner_point] += round(1-alpha, 4)
        if tmp_rank == rank:
            print ("out" + str(iter_index))
            break
        rank = tmp_rank
    right_num = 0
    for zuhe in sorted(rank.items(), key = operator.itemgetter(1), reverse=True):
        point, pr_score = zuhe[0], zuhe[1]
        if len(point.split('_')) < 2:
            continue
        if point in graph[root]:
            continue
        recom_result[point] = round(pr_score,4)
        right_num += 1
        if right_num > recom_num:
            break
    return recom_result


def get_one_user_recom():
    user='2'
    alpha=0.6
    graph=read.get_graph_from_data('../data/ratings.txt')
    iter_num=100
    recom_result=personal_rank(graph,user,alpha,iter_num,100)

    return recom_result
'''
    item_info=read.get_item_info('../data/movies.txt')
    for itemid in graph[user]:
        pure_itemid=itemid.split('_')[1]
        print(item_info[pure_itemid])
    print('result---------')
    for itemid in recom_result:
        pure_itemid=itemid.split('_')[1]
        print(item_info[pure_itemid])
        print(recom_result[itemid])
'''

def personal_rank_matrix(graph,root,alpha,recom_num=10):
    '''
    推荐产品给用户
    :param graph: 二分图
    :param root: fix user to recom
    :param alpha: the prob to random walk
    :param recom_num:
    :return: a dict,key:itemid,value:pr score
    A*r=r0
    '''
    m, vertex, address_dict =mat_util.graph_to_m (graph)
    if root not in address_dict:
        return {}
    score_dict={}
    recom_dict={}
    mat_all=mat_util.mat_all_point(m,vertex,alpha)   #A矩阵
    index=address_dict[root]
    initial_list=[[0] for row in range(len(vertex))]
    initial_list[index]=[1]
    r_zero=np.array(initial_list)
    res=gmres(mat_all,r_zero,tol=1e-8)[0]     #求解方程组,res就是r,n维向量，代表每个节点的PR，即重要度
    for index in range(len(res)):
        point=list(vertex)[index]
        if len(point.strip().split('_'))<2:
            continue
        if point in graph[root]:
            continue
        score_dict[point]=round(res[index],3)
    for combination in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:recom_num]:
        point,score=combination[0],combination[1]
        recom_dict[point]=score
    return recom_dict

def get_one_user_by_matrix():
    user='2'
    alpha=0.8
    graph = read.get_graph_from_data ('../data/ratings.txt')
    recom_result=personal_rank_matrix(graph,user,alpha,100)
    return recom_result


if __name__=='__main__':
    item_info = read.get_item_info ('../data/movies.txt')
    recom_result_base=get_one_user_recom()
    recom_result_matrix = get_one_user_by_matrix ()
    #测试部分1：观察推荐内容
    '''
    for itemid in recom_result_base:
        pure_itemid=itemid.split('_')[1]
        print(item_info[pure_itemid])
        print(recom_result_base[itemid])
    print('----------------')
    
    for itemid in recom_result_matrix:
        pure_itemid=itemid.split('_')[1]
        print(item_info[pure_itemid])
        print(recom_result_matrix[itemid])
    '''
    #测试部分2:两种方案推荐10条有9条相同，推荐100条有72条相同
    num=0
    for ele in recom_result_base:
        if ele in recom_result_matrix:
            num+=1
    print(num)










