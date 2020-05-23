from HERMES import ProblemSet, KP
import numpy as np
from tensorflow.keras.metrics import Mean
from tensorflow.python.ops import array_ops
import tensorflow.keras.backend as k
import tensorflow as tf

def predictionSet(instances, features, heuristics):
    data = np.empty((instances.getSize(), len(features)))
    # labels = np.empty(instances.getSize(), dtype=np.int8)
    profits = np.empty((instances.getSize(),len(heuristics)))
    for i_idx,i in enumerate(instances.getFiles()):
        p = KP(i)
        for f_idx,f in enumerate(features):
            data[i_idx][f_idx] = p.getFeature(f)
        for h_idx, h in enumerate(heuristics):
            p = KP(i)
            p.solve(h)
            profits[i_idx][h_idx] = p.getObjValue()
    labels = profits.argmin(axis=1)
    return (data, labels, profits)

class DBM(Mean):
    def __init__(self, name='DBM', **kwargs):
        super(DBM, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        row = tf.range(0, tf.shape(y_true)[0], dtype=tf.int64)
        col = k.argmin(y_pred, axis=1)
        pred_val = tf.gather_nd(y_true,tf.stack([row,col], axis=1))
        values = pred_val - k.min(y_true, axis=1)
        # tf.print(tf.gather(y_true,0), tf.gather(y_pred,0), tf.gather(pred_val,0),k.min(y_true, axis=1), values, sep='\n')
        return super(DBM, self).update_state(values, sample_weight=sample_weight)

def bayesian(profits):
    l = []
    for hp in profits.T:
        l.append(tf.reduce_mean(tf.abs(k.min(profits, axis=1) - hp)).numpy())
    return l