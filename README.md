# DecisionTree

Python决策树

## ID3实现

首先考虑决策树算法的原理：

通过**信息增益**的方式选出树的结点

所以我们需要知道什么是信息增益，或者只了解信息增益怎么算：

信息增益（H） = 信息熵（总） - 信息熵（某个属性）

信息熵（总） = -所有（分类的概率\*log2（分类的概率））之和
实现：
```python
def calcTotolEnt(dataSet):
    totolNum = len(dataSet)
    labelCounts = {}
    for line in dataSet:
        currentLabel = line[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    totolEnt = 0
    for key in labelCounts:
        probability = float(labelCounts[key])/totolNum
        totolEnt -= probability*log(probability,2)
    return totolEnt
```

信息熵（某个属性） = -所有（根据该属性分类的概率\*log2（根据该属性分类的概率））之和

在完成选出最优`属性`之前需要为后续将数据集分开做一个准备：
```python
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for line in dataSet:
        if line[axis] == value:
            reducedLine = line[:axis]
            reducedLine.extend(line[axis+1:])
            retDataSet.append(reducedLine)
    return retDataSet
```

