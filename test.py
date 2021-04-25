from sklearn.ensemble import RandomForestClassifier
import math


domainlist = []
clf = RandomForestClassifier(random_state=0)


def entropy(self):
    dict = {}
    for i in range(0, self.len):
        if self.name1[i] in dict.keys():
            dict[self.name1[i]] = dict[self.name1[i]] + 1
        else:
            dict[self.name1[i]] = 1
    entropy = 0
    for i in dict.keys():
        p = float(dict[i]) / self.len
        entropy = entropy - p * math.log(p, 2)
    return entropy


def number(self):
    number = 0
    for i in self.name1:
        if i.isdigit():
            number = number + 1
    return number


class Domain:
    def __init__(self, _name, _label=""):
        self.name1 = _name.split(".")[0]
        self.name2 = _name.split(".")[1]
        self.label = _label
        self.len = len(self.name1)

    def returnData(self):
        return [self.len, entropy(self), number(self)]

    def returnLabel(self):
        if self.label == "dga":
            return 0
        else:
            return 1


def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            domainlist.append(Domain(name, label))


def predict(readfname, writefname):
    with open(readfname, "r") as f1:
        with open(writefname, "w") as f2:
            for line in f1:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                ob = Domain(line)
                label = clf.predict([ob.returnData()])
                if label == 0:
                    f2.write(line+",dga\n")
                else:
                    f2.write(line+",notdga\n")


def main():
    print("Initialize Raw Objects")
    initData("train.txt")
    featureMatrix = []
    labelList = []
    print("Initialize Matrix")
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    print("Begin Training")
    clf.fit(featureMatrix, labelList)
    print("Begin Predicting")
    predict("test.txt", "result.txt")


if __name__ == '__main__':
    main()
