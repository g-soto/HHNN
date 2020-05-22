from jnius import java_method,PythonJavaClass
from HERMES import ProblemSet

class NNHH(PythonJavaClass):
    __javaclass__ = 'mx/tec/hermes/problems/ProblemSet'
    __javainterfaces__ = []

    # @java_method('()Ljava/util/List;')
    # def getFiles(self):
    #     print('ya sabía yo')
    #     return super().getFiles()
