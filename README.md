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

选择最好的属性去分类：
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