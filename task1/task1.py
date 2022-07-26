#   中山大学  2018级  软件工程
#   杨玲  2021年10月

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from minepy import MINE
from prettytable import PrettyTable

#特征名称
feature_head = ['max_degree','fail_node_degree','fail_neber_degree','fail_degree_sum',
                     'max_load','big_load_num','fail_load_sum','fail_num',
                     'first_round_fail','neber_fail_num','fail_round','subgraph_num',
                     'fail_node_load','load_change','degree_change','fail_neber_load']
number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

#按照得到结果进行排序
def sort_func(data, result):
    for i in range(data.shape[0]):
        count = 0
        for j in range(data.shape[0]):
            if(data[j] > data[i]):
                count = count + 1
        result[i] = count + 1


#卡方检验
def chi2_func(features, label, order):
    print("\n卡方过滤计算结果:")
    result1 = SelectKBest(chi2, k = 16)  # 选择K个最好的特征，返回选择特征后的数据
    result1.fit_transform(features, label)

    # 结果表格
    result_table = PrettyTable()
    result_table.add_column("Number", number)
    result_table.add_column("Feature Name", feature_head)
    result_table.add_column("Scores", result1.scores_)
    result_table.add_column("p-values", result1.pvalues_)
    scores = result1.scores_.reshape(16,1)
    sort_func(scores, order)
    result_table.add_column("Order", order)
    print(result_table)

#互信息
def mutual_info_func(features, label, order):
    print("\n互信息分类计算结果:")
    result2 = mutual_info_classif(features, label)

    # 结果表格
    result_table = PrettyTable()
    result_table.add_column("Number", number)
    result_table.add_column("Feature Name", feature_head)
    result_table.add_column("Scores", result2)
    scores = result2.reshape(16, 1)
    sort_func(scores, order)
    result_table.add_column("Order", order)
    print(result_table)

#标准化互信息
def normalized_mutual_info_func(features, label, order):
    print("\n标准化互信息分类计算结果:")
    i = 0
    result_NMI = np.zeros(16)
    for index, row in features.iteritems():
        result_NMI[i] = metrics.normalized_mutual_info_score(features[index], label)
        i = i + 1

    # 结果表格
    result_table = PrettyTable()
    result_table.add_column("Number", number)
    result_table.add_column("Feature Name", feature_head)
    result_table.add_column("Scores", result_NMI)
    sort_func(result_NMI, order)
    result_table.add_column("Order", order)
    print(result_table)

#最大互信息系数MIC
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)  # 选择 K 个最好的特征，返回特征选择后的数据

def mic_func(features, label, order):
    print("\n最大互信息系数(MIC)计算结果:")
    mic_select = SelectKBest(lambda X, y: tuple(map(tuple, np.array(list(map(lambda x: mic(x, y), X.T))).T)), k=16)
    mic_select.fit_transform(features, label)  #选择K个最好的特征，返回特征选择后的数据
    mic_scores = mic_select.scores_  # 特征与最大信息系数的对应

    #结果表格
    print("\n最大互信息系数(MIC)计算结果:")
    result_table = PrettyTable()
    result_table.add_column("Number", number)
    result_table.add_column("Feature Name", feature_head)
    result_table.add_column("Scores", mic_scores)
    sort_func(mic_scores, order)
    result_table.add_column("Order", order)
    print(result_table)

def main_func():
    # 导入数据集
    data = pd.read_table('2-AS.txt',index_col=False)
    print("\n源数据信息：")
    print("行数：", data.shape[0]) # 不包括表头
    print("列数：", data.columns.size)
    print(data)

    print("\n特征数据：")
    features = data[feature_head]
    print(features)

    print("\n标签数据：")
    label = data["LCC"]
    print(label)

    #卡方分布
    order1 = np.zeros(16, 'i')
    chi2_func(features, label, order1)

    #互信息
    order2 = np.zeros(16, 'i')
    mutual_info_func(features, label, order2)

    #标准化互信息
    order3 = np.zeros(16, 'i')
    normalized_mutual_info_func(features, label, order3)

    #最大互信息系数(MIC)
    order4 = np.zeros(16, 'i')
    mic_func(features, label, order4)

    print("\n汇总结果：")
    result_table = PrettyTable()
    result_table.add_column("Number", number)
    result_table.add_column("Feature Name", feature_head)
    result_table.add_column("chi2", order1)
    result_table.add_column("mutual_info", order2)
    result_table.add_column("normalized_mutual_info", order3)
    result_table.add_column("Maximal Information Coefficient(MIC)", order4)
    print(result_table)

main_func()