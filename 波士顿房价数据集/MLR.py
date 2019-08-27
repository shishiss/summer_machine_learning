import numpy as np
from regression import linear_regression 
from predict import predict
'''
获取波士顿房价数据集/housing_data.txt的内容,
并且返回list类型的训练数据集和测试数据集（尚未分为特征和标签）
'''
def get_Data(filename, split):
	train_data = []
	test_data = []
	with open(filename) as txtData:
		lines = txtData.readlines()
		count = 0 #记录当前行数
		for line in lines:
			line = line.strip().split()#在这儿犯了愚蠢的错误，将line一行的内容作为一个数据了
			if count < split:
				train_data.append(line)
				count = count + 1
			else:
				test_data.append(line)
				count = count + 1

	#将字符串数据全部转换成float
	
	for i in range(0,len(train_data)):
		for j in range(0, len(train_data[0])):
			train_data[i][j] = float(train_data[i][j])
		#del(train_data[i][0:len(train_data[0])])

	for i in range(0, len(test_data)):
		for j in range(0, len(test_data[0])):
			test_data[i][j] = float(test_data[i][j])
		#del(test_data[i][0:len(test_data[0])])

	return train_data, test_data


'''
输入为list类型的数据，以下操作是将数据分为特征和标签两部分，
返回为np.array类型的特征数组和标签数组
'''
def split_data(data_list):
	character = []
	label = []
	for i in range(len(data_list)):
		#print(data_list[i][0:-2])
		character.append(data_list[i][:-1])
		label.append(data_list[i][-1])
	return np.array(character), np.array(label)


if __name__ == '__main__':
	file_path = '/Users/yuqishi/Documents/machine_learning/data/波士顿房价数据集/housing_data.txt'
	split = 450

	#获得训练样本和测试样本
	train_data, test_data = get_Data(file_path, split)

	#获得训练样本的特征和标签数组
	train_X, train_Y = split_data(train_data)

	#获得测试数组X的特征和标签数组
	test_X, test_Y = split_data(test_data)

	#将训练集中的数据训练求得w参数值
	w = linear_regression(train_X, train_Y)
	print(predict(test_X, test_Y, w))








