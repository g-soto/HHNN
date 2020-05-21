import os
import jnius_config
jnius_config.set_classpath(os.path.dirname(os.path.realpath(__file__)) + '\\HERMES.jar')

from jnius import autoclass

__all__ = ['KP', 'ProblemSet', 'SubSet', 'RuleBasedHH']

KP = autoclass('mx.tec.knapsack.problem.KP')
ProblemSet = autoclass('mx.tec.hermes.problems.ProblemSet')
SubSet = autoclass('mx.tec.hermes.problems.ProblemSet$Subset')
RuleBasedHH = autoclass('mx.tec.hermes.frameworks.rulebased.RuleBasedHH')
String = autoclass('java.lang.String')



