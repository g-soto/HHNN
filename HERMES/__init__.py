import os
import jnius_config
# jnius_config.set_classpath('"' + os.path.dirname(os.path.realpath(__file__)) + '\\HERMES.jar"')
jnius_config.set_classpath(os.path.dirname(os.path.realpath(__file__)) + '\\HERMES.jar;' 
                         + os.path.dirname(os.path.realpath(__file__)) + '\\extra.jar')

# print("'" + os.path.dirname(os.path.realpath(__file__)) + '\\HERMES.jar' + "'")
from jnius import autoclass

__all__ = ['KP', 'ProblemSet', 'SubSet', 'NNHH']

KP = autoclass('mx.tec.knapsack.problem.KP')
ProblemSet = autoclass('mx.tec.hermes.problems.ProblemSet')
SubSet = autoclass('mx.tec.hermes.problems.ProblemSet$Subset')
NNHH = autoclass('NN.NNHH')
