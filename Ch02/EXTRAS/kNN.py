from numpy import zeros
def file2matrix(filename):
    f = open(filename)
    lines = f.readlines()
    numberOfLins = len(lines)
    mat = zeros((numberOfLins, 3))
    lablevector = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        mat[index, :] = listFromLine[0:3]
        lablevector.append(int(listFromLine[-1]))
        index += 1

    return mat, lablevector