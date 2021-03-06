from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import math

def load_dataset(dataset_path, n_train_data):
    """加载数据集，对数据进行预处理，并划分训练集和验证集
    :param dataset_path: 数据集文件路径
    :param n_train_data: 训练集的数据量
    :return: 划分好的训练集和验证集
    """
    dataset = []
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(dataset_path, 'r') as file:
        # 读取CSV文件，以逗号为分隔符
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            # 将字符串类型的特征值转换为浮点型
            row[0:4] = list(map(float, row[0:4]))
            # 将标签替换为整型
            row[4] = label_dict[row[4]]
            # 将处理好的数据加入数据集中
            dataset.append(row)

    # 对数据进行归一化处理
    dataset = np.array(dataset)
    mms = MinMaxScaler()
    for i in range(dataset.shape[1] - 1):
        dataset[:, i] = mms.fit_transform(dataset[:, i].reshape(-1, 1)).flatten()

    # 将类标转为整型
    dataset = dataset.tolist()
    for row in dataset:
        row[4] = int(row[4])
    # 打乱数据集
    random.shuffle(dataset)

    # 划分训练集和验证集
    train_data = dataset[0:n_train_data]
    val_data = dataset[n_train_data:]

    return train_data, val_data


def initializze_network(n_inputs,n_outputs):
    """
    初始化竞争神经网络
    :param n_inputs: 每个输入的属性数量
    :param n_outputs: 分类的类别数量
    :return: 初始化后的神经网络
    """
    network = list()
    #竞争层
    #该列表有n_outputs个字典。每个字典大小为n_inputs。每个字典的意义是输入层每个属性到该神经元的权重。最后一个权重为偏置权重
    output_layer = [{"weights":[0.5 for i in range(n_inputs+1)]}for j in range(n_outputs)]
    network.append(output_layer)
    return network

def train(train_data,l_rate,epochs,val_data):
    """
    :param train_data: 训练集
    :param l_rate: 学习速率
    :param epochs: 迭代回合数
    :param val_data: 验证集
    :return: 训练好的网络
    """
    #获取特征列数
    n_inputs =len(train_data[0])-1
    #获取分类的总数
    n_outputs = len(set([row[-1] for row in train_data]))
    #初始化神经网络
    network = initializze_network(n_inputs,n_outputs)
    for i in range(n_outputs):#加入计数
        for j in range(n_outputs):
            network[0][i][j] = 0
    acc = [] #准确率数组
    for epoch in range (epochs):#训练epochs个回合
        for row in train_data:
            # 找到胜者ID ，在这里为 0,1,2
            winnerId = findWinner(row,network[0])
            weights = network[0][winnerId]["weights"]
            for i in range(len(weights)-1): #注意最后一个是偏置的权值,在这里不用管
                weights[i] = weights[i] + l_rate*(row[i]-weights[i])
            network[0][winnerId]["weights"] = weights
            network[0][winnerId][row[-1]] += 1
        acc.append(validation(network, val_data))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(acc)
    plt.show()
    return network,acc[-1]


def findWinner(row,comLayer):
    """
    找到距离该行距离最近的神经元，返回它的下标
    :param row: 某一行数据
    :param comLayer: 竞争层
    :return:
    """
    distance = [] #每个竞争神经元和权值之间的距离
    for neuron in comLayer:
        weights = neuron["weights"] #该神经元的权值
        dis = 0.0
        for i in range(len(weights)):
            dis += pow(row[i]-weights[i],2)
        distance.append(math.sqrt(dis))
    winnerId = distance.index(min(distance))
    return winnerId



def predict(network,row):
    """

    :param network:
    :param row:
    :return:
    """
    outputs = process(network,row)
    maxNeuron = outputs.index(max(outputs))  #最大神经元下标
    n_outputs = len(network[0])              #输入的类别数
    index = 0  #该神经元所代表的类别
    tmp = 0    #最大值
    for i in range(n_outputs): #找出该神经元对应的类别编号
        if network[0][maxNeuron][i]>tmp:
            tmp = network[0][maxNeuron][i]
            index = i
    return index


def process(network,inputs):
    """
    计算输出
    :param network:神经网络
    :param inputs:输入(已经归一化)
    :return: 各个神经元输出
    """
    outputs = []
    output_layer = network[0]
    for neuron in output_layer:
        weights = neuron["weights"]
        output = 0.0
        for i in range(len(weights)-1):
            output += inputs[i]*weights[i]
        # output += 1*weights[-1] #偏置为1，权重为neuron[-1]
        outputs.append(output)
    return outputs


def validation(network,val_data):
    """
    测试神经网络在验证集上的效果
    :param network: 神经网络
    :param val_data: 验证集
    :param n_outputs: 类别数量
    :return: 模型在验证集上的准确率
    """
    n_outputs = len(network[0])
    # 获取预测类标
    predicted_label = []
    for row in val_data:
        prediction = predict(network, row)
        index = 0 #类别
        tmp = -1
        for i in range (n_outputs):#得到这名神经元的类别
            if network[0][prediction][i]>tmp:
                tmp = network[0][prediction][i]
                index = i
        predicted_label.append(i)
    # 获取真实类标
    actual_label = [row[-1] for row in val_data]
    # 计算准确率
    accuracy = accuracy_calculation(actual_label, predicted_label)
    # print("测试集实际类标：", actual_label)
    # print("测试集上的预测类标：", predicted_label)
    return accuracy


def accuracy_calculation(actual_label, predicted_label):
    """计算准确率
    :param actual_label: 真实类标
    :param predicted_label: 模型预测的类标
    :return: 准确率（百分制）
    """
    correct_count = 0
    for i in range(len(actual_label)):
        if actual_label[i] == predicted_label[i]:
            correct_count += 1
    return correct_count / float(len(actual_label)) * 100.0


if __name__ == "__main__":
    file_path = './IrisDataset/iris.csv'
    l_rate = 0.1  # 学习率
    epochs = 1000  # 迭代训练的次数
    n_train_data = 120  # 训练集的大小
    train_data, val_data = load_dataset(file_path, n_train_data)
    network, acc = train(train_data, l_rate, epochs, val_data)
    print("准确率",acc)
    index = predict(network,[5.9,3.0,5.1,1.8])
    print("分类：",index)