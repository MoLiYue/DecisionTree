# DecisionTree

Python决策树

## ID3实现

首先考虑决策树算法的原理：

通过**信息增益**的方式选出树的结点

所以我们需要知道什么是信息增益，或者只了解信息增益怎么算：

信息增益（H） = 信息熵（总） - 信息熵（某个属性）

信息熵（总） = -所有（**不同结论**的概率\*log2（**不同结论**的概率））之和
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

信息熵（某个属性） = -所有（*根据该属性分类后***不同结论**的概率\*log2（*根据该属性分类后***不同结论**的概率））之和

由于需要计算`根据该属性分类后`的不同结论的概率，我们需要获得`根据该属性分类`的子集：
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
获得子集时去掉了已经分过类的属性。

选择出最好的属性并依据该属性进行分组：
```python
def chooseBestFeatureToSplit(dataSet):
    featureNumbers = len(dataSet[0])-1
    baseEntropy = calcTotolEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for featureNO in range(featureNumbers):
        featureList = [line[i] for line in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(data,featureNO,value)
            probability = len(subDataSet)/float(len(dataSet))
            newEntropy += probability*calcTotolEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = featureNO
    return bestFeature
```

考虑依照大多数原则，我们需要确定每种分类下的最终属性

每种分类下数量最多的属性被认为是该分类下的最终属性
```python
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = -0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
```

做好准备后可以开始创建树：

创建树使用递归创建

结束条件为：

1.在某一属性的分类下，所有的结论都相同

2.只剩最后一列属性

创建树，使用递归

```python
def createTree(dataSet,labels):
    classList = [line[-1] for line in dataSet]
    if classList.count (classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeatureColumNum = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeatureColumNum]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeatureColumNum])
    featValus = [line[bestFeatureColumNum] for line in dataSet]
    uniqueValus = set(featValus)
    for value in uniqueValus:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                            (dataSet,bestFeatureColumNum,value),subLabels)
    return myTree
```

运行程序