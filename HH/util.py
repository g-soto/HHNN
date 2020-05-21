from HERMES import ProblemSet, KP, String
import numpy as np

def predictionSet(instances, features, heuristics):
    data = np.empty((instances.getSize(), len(features)))
    labels = np.empty(instances.getSize(), dtype=np.int8)
    profits = np.empty(len(heuristics))
    for i_idx,i in enumerate(instances.getFiles()):
        p = KP(i)
        for f_idx,f in enumerate(features):
            data[i_idx][f_idx] = p.getFeature(f)
        for h_idx, h in enumerate(heuristics):
            p = KP(i)
            p.solve(h)
            profits[h_idx] = p.getObjValue()
        labels[i_idx] = profits.argmin()
    return (data, labels)
    