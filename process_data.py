from sche_interfere import Sche_interfere
import torch
import math

class Procdata:

    def make_edge_T_C(self, T_C, numtasks, tasks_to_gen):

        sche_interfere = Sche_interfere()
        V_Interfere = torch.empty(tasks_to_gen, 4)
        T_C_Interfere = torch.zeros(numtasks, numtasks)

        for iterate_sets in range(tasks_to_gen):
            V_Interfere[iterate_sets][0] = sche_interfere.V1(T_C[iterate_sets][0], T_C[iterate_sets][1])
            V_Interfere[iterate_sets][1] = sche_interfere.V2(T_C[iterate_sets][0], T_C[iterate_sets][1])
            V_Interfere[iterate_sets][2] = sche_interfere.V3(T_C[iterate_sets][0], T_C[iterate_sets][1])
            V_Interfere[iterate_sets][3] = sche_interfere.V4(T_C[iterate_sets][0], T_C[iterate_sets][1])



        for i in range(tasks_to_gen):
            for k in range(i+1, tasks_to_gen):
                T_C_Interfere[i][k] = (V_Interfere[k][0] * V_Interfere[i][2] +  V_Interfere[k][1] * V_Interfere[i][3])*1000


        return T_C_Interfere



    def make_edge_uniproc_T_C(self, T_C, numtasks, tasks_to_gen):

        #T_C_Interfere = torch.zeros(numtasks, numtasks)

        #for i in range(tasks_to_gen):
        #    for k in range(i+1, tasks_to_gen):
        #        T_C_Interfere[i][k] = (math.ceil(T_C[k][0]/T_C[i][0])*T_C[i][1])/T_C[k][0]
        T_C_Interfere = []
        for i in range(tasks_to_gen):
            for k in range(i+1, tasks_to_gen):
                T_C_Interfere.append((math.ceil(T_C[k][0]/T_C[i][0])*T_C[i][1])/T_C[k][0])
                #T_C_Interfere[i][k] = (math.ceil(T_C[k][0]/T_C[i][0])*T_C[i][1])/T_C[k][0]

        T_C_Interfere = torch.tensor(T_C_Interfere)

        return T_C_Interfere
