from random import randrange
from operator import itemgetter

class GenSet:

    def makeTC(self, numtasks, num_processor, maxT):
        T_C = []
        for i in range(numtasks):
            T = randrange(2, maxT)
            C = randrange(1, T)
            T_C.append([T, C])

        return T_C

    def rate_monotonic(self, inputlist):
        inputlist = sorted(inputlist, key=itemgetter(0), reverse=False)

        return inputlist

    def makeTC_binomial(self, numtasks, num_processor, maxT):
        T_C = []

        for i in range(int(numtasks*0.9)):
            T = randrange(4, maxT)
            C = randrange(1, int(T*0.5))
            T_C.append([T, C])

        for i in range(numtasks - int(numtasks*0.9)):
            T = randrange(2, maxT)
            C = randrange(1, T)
            T_C.append([T, C])

        return T_C
