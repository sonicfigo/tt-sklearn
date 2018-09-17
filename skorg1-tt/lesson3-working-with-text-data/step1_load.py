# coding=utf-8
"""
Loading the 20 newsgroups dataset
The dataset is called “Twenty Newsgroups”.

加载数据的简单操作
"""
from datasets import load_train

twenty_train = load_train()

print(twenty_train.target_names)  # 4个类型

print('\n===================data：X,2257 份 data(一份data就是一份新闻内容)')
data_list = twenty_train.data
print(len(data_list))
print(data_list[0])

print('\n===================targetname: 4种新闻类型')
target_name_list = twenty_train.target_names
print(target_name_list)  # 4个

print('\n===================filenames：2257个文件名')
file_names = twenty_train.filenames  # numpy.ndarray
print(len(file_names))  # 2257 个文件名
print(file_names[0])
print(file_names.shape)  # (2257, )

print('\n===================target：2257个答案index')
y_train = twenty_train.target
print(len(y_train))
print(y_train)
print(y_train.shape)  # (2257,)

print('\n===================y_train第一个答案index，对应的文字')
print(target_name_list[y_train[0]])
