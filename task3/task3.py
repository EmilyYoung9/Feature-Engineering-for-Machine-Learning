#   中山大学  2018级  软件工程
#   杨玲  2021年12月

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  # 引入KNN分类器
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def main_func():
    # 导入数据集
    data = pd.read_csv("2-USAirlines_cla.txt", sep='\t', index_col=False)
    print("\n源数据信息：")
    print(data)

    label = data.iloc[:, 17]
    print("\n标签数据：")
    print(label)

    accuracy_score_list = []  # 准确率
    for number in range (2,18):
        print("\n特征数据：",number - 1,"个")
        features = data.iloc[:, 1:number]
        # print(features)

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

        # print("\n训练集特征数据：")
        # print(train_data_feature)
        # print("\n训练集标签数据：")
        # print(train_data_label)
        # print("\n测试集特征数据：")
        # print(test_data_feature)
        # print("\n测试集标签数据：")
        # print(test_data_label)

       #KNN
        knn = KNeighborsClassifier()  # 调用KNN分类器
        knn.fit(train_data_feature.values, train_data_label.values)  # 训练KNN分类器
        test_pred = knn.predict(test_data_feature)
        score = accuracy_score(test_pred, test_data_label)
        accuracy_score_list.append(score)

    feature_number = []  # 序号
    for i in range(16):
        feature_number.append(i + 1)

    print("\n最终结果：")
    result_table = PrettyTable()
    result_table.add_column("Feature Number", feature_number)
    result_table.add_column("Accuracy Score", accuracy_score_list)
    print(result_table)

    plt.figure()
    plt.plot(feature_number, accuracy_score_list, 'x-', c='b', linewidth=1, label="accuracy score")
    plt.title("2-USAirlines_cla.txt")
    plt.xlabel("Feature Number")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.show()

main_func()