#-*-coding:utf8-*-
"""
author:jacquelin
date:201903
mat util for personal rank algo
矩阵算法
"""

from __future__ import division
from scipy.sparse import coo_matrix
import numpy as np
import util.read as read
import sys

def graph_to_m(graph):
    '''
    形成矩阵算法中的m,此处建议用jupyter notebook显示,更容易理解
    :param graph: 之前得到的二分图
    :return:
        a coo_matrix, sparse mat M
        a list, total user item point 所有用户和物品的顶点
        a dict, map all the point to row index
    '''
    vertex=graph.keys()   #所有顶点
    address_dict={}
    total_len=len(vertex)  #函数得出的是一个方阵，用示例数据log看，有9个顶点，方针就是9*9，total_len=9
    for index in range(len(vertex)):
        address_dict[list(graph.keys())[index]]=index
    row=[]
    col=[]
    data=[]
    #graph_log:{'A': {'item_a': 1, 'item_b': 1, 'item_d': 1}, 'item_a': {'A': 1, 'B': 1}, 'item_b': {'A': 1, 'C': 1}, 'item_d': {'A': 1, 'D': 1}, 'B': {'item_a': 1, 'item_c': 1}, 'item_c': {'B': 1, 'D': 1}, 'C': {'item_b': 1, 'item_e': 1}, 'item_e': {'C': 1}, 'D': {'item_c': 1, 'item_d': 1}}
    for element_i in graph:
        weight=round(1/len(graph[element_i]),3)  #出度分之一
        row_index=address_dict[element_i]
        for element_j in graph[element_i]:
            col_index=address_dict[element_j]
            row.append (row_index)
            col.append (col_index)
            data.append(weight)
    row=np.array(row)
    col = np.array (col)
    data=np.array(data)
    m=coo_matrix((data,(row,col)),shape=(total_len,total_len))
    return m,vertex,address_dict


def mat_all_point(m_mat,vertex,alpha):
    '''
    get E-alpha*m_mat.T,矩阵公司逆里边的部分
    :param m_mat: 上面函数得出的m
    :param vertex:所有顶点
    :param alpha:
    :return:
       a sparse稀疏矩阵
    '''
    total_len=len(vertex)
    row=[]
    col=[]
    data=[]
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    eye_t=coo_matrix((data,(row,col)),shape=(total_len,total_len))
    #print(eye_t.todense())
    #sys.exit()
    return eye_t.tocsr()-alpha*m_mat.tocsr().transpose()


if __name__=='__main__':
    graph=read.get_graph_from_data('../data/log.txt')
    m,vertex,address_dict=graph_to_m(graph)
    #print(mat_all_point(m,vertex,0.8))
    mat_all_point (m, vertex, 0.8)






