import math

class Sche_interfere:
    
    def V1(self, T, C):
        return T / (T-C)
    
    def V2(self, T, C):
        return 1 / (T-C)

    def V3(self, T, C):
        return C/T
    
    def V4(self, T, C):
        return C
