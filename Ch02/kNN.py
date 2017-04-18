# coding: utf-8
'''

Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    '''
    kNN分类器，根据已知分类(数据对应标签)数据判断末知分类数据类型.
    :param inX:     末分类数据
    :param dataSet: 已分类数据
    :param labels:  已分类数据的标签
    :param k:       k个附近值
    :return:        返回最接近的类型
    '''
    dataSetSize = dataSet.shape[0]
    # print '已知数据的行数:%d' %dataSetSize
    # print  '被测数据的行列数:', inX.shape

    # 把被测数据按已知数据的行数扩充
    # 思路：已知数据为n行， 被测数据要扩充成n份(行)
    # 然后求两者差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # 求差的平方
    sqDiffMat = diffMat**2
    # axis=1要按行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 1/2次方为开平方根
    distances = sqDistances**0.5
    # 递增加列序，并取得索引值
    sortedDistIndicies = distances.argsort()
    # 标签0：次数, ... ..., 标签n:次数
    classCount={}
    # 统计符合标签的被测数据出现的次数
    for i in range(k):
        # 取前k个数据标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 累计标签值相同的次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 标签排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最高的
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    '''
    打开数据文件
    :param filename:
    :return:
    '''
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    # 创建文件行数x3列的空数组:
    # 每行为:飞行里程数 游戏时间百分比 每周冰其淋数
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    # 对应的喜欢程度标签
    classLabelVector = []                       #prepare labels return
    # 打开数据文件
    fr = open(filename)
    index = 0
    # line = 飞行里程数 游戏时间百分比 每周冰其淋数  标签
    # 标签： 不喜欢/一般/有魅力
    for line in fr.readlines():
        # 删除回车符
        line = line.strip()
        # 以制表符切分
        listFromLine = line.split('\t')
        # 飞行里程数 游戏时间百分比 每周冰其淋数 为一行数组
        # 这里str类型直接赋给ndarray类型，会直动转成float类型
        returnMat[index,:] = listFromLine[0:3]
        # 对应的标签数值
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    print type(returnMat[0][0])
    return returnMat,classLabelVector

    
def autoNorm(dataSet):
    '''
    数据归一化， 即把数据转换成0-1的范围，可以根据反馈值还原到原始值
    公式： newValuw = (newValue-min)/(max-min)
    :param dataSet: 原始数据
    :return: 归一化值， 范围， 最小值
    '''
    # 取列的最小值(1行):[MIN0, ..., MINn]
    minVals = dataSet.min(0)
    # 取列的最大值(1行):MAX0, ..., MAXn]
    maxVals = dataSet.max(0)
    # 归一化范围
    ranges = maxVals - minVals
    # 创建一个维度和原来样的数组:n行xm列
    normDataSet = zeros(shape(dataSet))
    # 取行数n
    n = dataSet.shape[0]
    # 先把1行 x m列的最小值扩充成n行,然后与原数求差
    normDataSet = dataSet - tile(minVals, (n,1))
    # 先把1行 x m列的范围扩充成n行,然后除差值
    normDataSet = normDataSet/tile(ranges, (n,1))   #element wise divide
    #返回：归一化值， 范围， 最小值
    return normDataSet, ranges, minVals
   
def datingClassTest():
    #取10的数据为测试用
    hoRatio = 0.10      #hold out 10%
    # 获取数据和标签
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    # 把数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 取行数
    m = normMat.shape[0]
    # 取10%行数做测试
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    #取numTestVecs行数做测试， m-numTestVecs作为已分类本样和标签
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

    
def img2vector(filename):
    '''
    把文本图像矩阵文件转成 1x1024向量
    :param filename: 文件名
    :return:
    '''
    # 创建shape为1x1024的数组
    returnVect = zeros((1,1024))
    # 读取文件并转成1x1024的数字数据
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest(path = './'):
    hwLabels = []
    # 获取文件名列表
    trainingFileList = listdir(path + 'trainingDigits')
    # 文件数量
    m = len(trainingFileList)
    # 创建一个m * 1024的数组，填充0
    trainingMat = zeros((m, 1024))
    print '文件数量', m,
    print '数组型状', trainingMat.shape

    # 提取训练数据把每个文件转成1x1024的向量
    # 提取文件名中的数字，fileName = '0_0.txt'
    # 第一个0代表txt内存的图片内空是0字
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 文件对应的数字值存放到Label里
        hwLabels.append(classNumStr)
        # 转换完的文件1x1024向量存成一行
        trainingMat[i,:] = img2vector(path + 'trainingDigits/%s' % fileNameStr)

    # 测试文件列表
    testFilelist = listdir(path + 'testDigits')
    # 错误计数
    errorCount = 0.0
    mTest = len(testFilelist)

    # 提取测试数据并使用kNN函数求得分类
    # 并比较结果后统计错误率
    for i in range(mTest):
        fileNameStr = testFilelist[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 转换完的文件1x1024向量存成一行
        vectorUnderTest = img2vector(path + 'testDigits/%s' % fileNameStr)
        # 调用分类器进行分类参数：被测数据数组、已知数据数组、已知数据对就标签（对应的归类）、K值
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 100)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        # 比较结果，统计错误次数
        if (classifierResult != classNumStr): errorCount += 1.0

    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))



if __name__ == '__main__':
    # handwritingClassTest('./digits/')
    datingClassTest()



