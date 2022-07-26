#   中山大学  2018级  软件工程
#   杨玲  2021年11月

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  # 引入KNN分类器
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable
import matplotlib.pyplot as plt

#特征名称
feature_head_16 = ['max_degree','fail_node_degree','fail_neber_degree','fail_degree_sum',
                     'max_load','big_load_num','fail_load_sum','fail_num',
                     'first_round_fail','neber_fail_num','fail_round','subgraph_num',
                     'fail_node_load','load_change','degree_change','fail_neber_load']
feature_head_12 = ['fail_neber_degree','fail_degree_sum',
                     'max_load','fail_load_sum','fail_num',
                     'first_round_fail','neber_fail_num','subgraph_num',
                     'fail_node_load','load_change','degree_change','fail_neber_load']
feature_head_8 = ['fail_degree_sum',
                     'fail_load_sum','fail_num',
                     'first_round_fail',
                     'fail_node_load','load_change','degree_change','fail_neber_load']
feature_head_4 = ['fail_load_sum',
                     'first_round_fail',
                     'load_change','fail_neber_load']

def main_func():
    # 导入数据集
    data = pd.read_table('1-AS_cla.txt', sep='\t', index_col=False)
    print("\n源数据信息：")
    print(data)
    print("\n特征数据：")
    features = data[feature_head_12]
    print(features)
    print("\n标签数据：")
    label = data["LCC"]
    print(label)

    train_data_feature = pd.DataFrame()
    train_data_label = pd.DataFrame()
    test_data_feature = pd.DataFrame()
    test_data_label = pd.DataFrame()

    # 将源数据分为训练集和测试集
    for i in range(features.shape[0]):
        temp = pd.DataFrame([label.loc[i]], columns=['LCC'])
        if ((i + 1) % 10 == 2):
            test_data_feature = test_data_feature.append(features.loc[i], ignore_index=True)
            test_data_label = test_data_label.append(temp, ignore_index=True)
        else:
            train_data_feature = train_data_feature.append(features.loc[i], ignore_index=True)
            train_data_label = train_data_label.append(temp, ignore_index=True)

    print("\n训练集特征数据：")
    print(train_data_feature)
    print("\n训练集标签数据：")
    print(train_data_label)
    print("\n测试集特征数据：")
    print(test_data_feature)
    print("\n测试集标签数据：")
    print(test_data_label)

    accuracy_score_list = []    #准确率
    number = []     #序号
    for i in range(20):
        number.append(i + 1)

    #KNN
    for k in range(20):
        knn = KNeighborsClassifier(n_neighbors = k + 1)  # 调用KNN分类器
        knn.fit(train_data_feature.values, train_data_label.values)  # 训练KNN分类器
        test_pred = knn.predict(test_data_feature)
        score = accuracy_score(test_pred, test_data_label)
        accuracy_score_list.append(score)


    print("\n最终结果：")
    result_table = PrettyTable()
    result_table.add_column("k", number)
    result_table.add_column("Accuracy Score", accuracy_score_list)
    print(result_table)

    plt.figure()
    plt.plot(number, accuracy_score_list, 'x-', c='b', linewidth=1, label="accuracy score")
    plt.title("Number of Features: 12")
    plt.xlabel("k")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.show()

main_func()