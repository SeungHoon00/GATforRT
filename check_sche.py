import numpy as np
import torch
import pickle
import gzip
from RM_RTA import RM_RTA

NUMPROC = 7

class Check_sche:
    def read_data(self):
        nbr_sets = 0
        success = 0
        
        rm_rta = RM_RTA()
        bookmark = 0
        with open('./Test/8proc_mixed_paper_features.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('./Test/8proc_mixed_paper_index.pkl', 'rb') as f:
            index = pickle.load(f)
        pre_T = 0

        cnt = 0
        bookmark = 0

        task_set = []
        for row in features:
            if pre_T <= row[0]:
                task_set.append(row[:2])
                pre_T = row[0]
                cnt += 1
            else:
                pre_T = 0
                unschedulable = rm_rta.rmrta(task_set, index[bookmark:bookmark+cnt], NUMPROC)
                    #print(task_set)
                if unschedulable == False:
                    success += 1
                    with open('./Findings/8proc_mixed_newsche_taskset.txt', 'a') as f:
                        f.write(str(task_set) + '\n')
                    with open('./Findings/8proc_mixed_newsche_index.txt', 'a') as f:
                        f.write(str(index[bookmark:bookmark+cnt])+ '\n')
                    print(task_set)
                    print(index[bookmark:bookmark+cnt])
                bookmark = bookmark+cnt
                nbr_sets += 1
                cnt = 0
                task_set = []
                task_set.append(row[:2])
                pre_T = row[0]
                cnt += 1
        
        print(nbr_sets+1)
        print(success)

if __name__ == '__main__':
    Check_sche().read_data()

