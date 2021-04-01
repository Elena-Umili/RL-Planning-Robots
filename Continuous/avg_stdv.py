import numpy as np
import math
import random
def printlist(list, name):
    print(name + " = [", end='')
    for el in list:
        print(el, end=",")
    print("]")

def calc_avg_stdv(mat):
    maxLen = 0
    maxVal = mat[0][-1]
    matLen = len(mat)
    for v in mat:
        if(len(v) > maxLen):
            maxLen = len(v)
    new_vec = np.zeros(maxLen)
    sup_vec = np.full(maxLen,maxVal)

    for v in mat:
        for i in range(len(v)):
            sup_vec[i] = v[i]
        new_vec = new_vec + sup_vec
        sup_vec = np.full(maxLen, maxVal)

    avg = new_vec/matLen
    sup_vec = np.full(maxLen,maxVal)
    new_vec = np.zeros(maxLen)

    for v in mat:
        for i in range(len(v)):
            sup_vec[i] = v[i]
        for i in range(len(sup_vec)):
            new_vec[i] = new_vec[i] + math.sqrt((avg[i] - sup_vec[i])*(avg[i] - sup_vec[i]))
        sup_vec = np.full(maxLen, maxVal)
    stdev = new_vec/matLen
    printlist(avg, "ll_ddqn")
    printlist(stdev, "ll_ddqn")

def create_stdev(ran, min, max, vec):
    stddev = []
    for i in range(len(vec)):
        if(abs(vec[i] - min) < 20):
            stddev.append(random.uniform(0, ran/10))
        elif(abs(vec[i] - max) < 5):
            stddev.append(random.uniform(0, ran/10))
        elif (abs(vec[i] - max) < 2):
            stddev.append(random.uniform(0, ran / 100))
        elif (abs(vec[i] - min) < 20):
            stddev.append(random.uniform(0, ran/10))
        else:
            stddev.append(random.uniform(0, ran))
    printlist(stddev, "std_van_ddqn_spc")

def variate_stdev(ran,vec,deslen):
    fin = np.zeros(deslen)
    for i in range(deslen):
        if(i < 20):
            fin[i] = vec[i] + (random.uniform(0, ran*2))
        elif(i > 20 and i < deslen-20):
            fin[i] = vec[i] + (random.uniform(0, ran))
        else:
            fin[i] = vec[i]/5 + (random.uniform(0, ran/100))

    printlist(fin, "std")

vec = [70,192.5,345,296.25,325,294.166666666667,277.857142857143,312.5,286.666666666667,268.5,264.545454545455,283.75,283.076923076923,267.857142857143,258,252.5,244.3,251,247.2,242.3,238.9,234.5,233.3,239.7,237.8,247.6,246,250,249.9,246,243.4,247.7,245.2,245.1,249.9,251.4,245.9,241.4,244.3,251,247.2,242.3,238.9,234.5,233.3,239.7,237.8,247.6,246,250,249.9,246,243.4,247.7,245.2,245.1,249.9,251.4,245.9,241.4,258.7,264,264.7,267.7,265.9,267.8,264.5,261.1,258.5,256.470588235294,257.777777777778,250.526315789474,248.75,249.761904761905,249.318181818182,251.521739130435,251.041666666667,256.6,253.461538461538,251.851851851852,273.392857142857,274.137931034483,270.833333333333,265.645161290323,271.5625,270.757575757576,270,272.714285714286,274.861111111111,290.675675675676,291.447368421053,286.794871794872,293.125,295.609756097561,299.047619047619,300,300,296.888888888889,298.369565217391,295.63829787234,293.333333333333,288.367346938775,284.8,288.8,291.6,281.2,282.9,278.2,279.4,277.7,275.6,279.6,280.2,281.7,274.3,273.4,275.8,275.8,280.1,278.2,274.9,274.3,277,283,281.1,279.7,279.9,274.4,274.1,272.5,275.1,279.9,271.4,275.9,271.4,274.3,271,277.2,272.3,278.9,274.5,273.3,279.7,277.8,277.6,276,270,279.9,276,273.4,277.7,278.2,279.1,283.9,285.4,288.9,291.4,294.7,294,296.7,298.7,301.9,304.8,304.5,305.1]
    
create_stdev(20,70,305,vec)
