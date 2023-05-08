from genset import GenSet
from process_data import Procdata
from genmask import GenMask
from FBBFFD import FBBFFD
from RM_RTA import RM_RTA
import numpy as np
import torch
import random


class Preprocessing:
    def gen_dataset(self, numtasks, num_processor, MAX_period, dataset_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generate_tasks = GenSet()
        procdata = Procdata()
        generate_mask = GenMask()
        heuristic_fbbffd = FBBFFD()
        rm_rta = RM_RTA()
        

        total_attr = []
        total_FBBFFD_sche = []
        total_edges = []
        total_weights = []

        for size in range(dataset_size):

            while True:
                taskset = generate_tasks.makeTC(numtasks, MAX_period)
                # Rate Monotonic
                RM_taskset = generate_tasks.rate_monotonic(taskset)
                ANS_heuristic, unschedulable_with_ANS = heuristic_fbbffd.fbbffd(RM_taskset, num_processor)


                if unschedulable_with_FBBFDD:
                    continue


                FBBFFD_sche = [999] * numtasks

                #print(RM_taskset)
                #print(FBBFFD_heuristic)
                for i in range(len(RM_taskset)):
                    for j in range(len(ANS_heuristic)):
                        for x in ANS_heuristic[j]:
                            if RM_taskset[i] == x:
                                FBBFFD_sche[i] = j
                                break


                attr = []
                for i in range(numtasks):
                    attr.append([RM_taskset[i][0], RM_taskset[i][1], RM_taskset[i][0], (RM_taskset[i][0]/RM_taskset[i][1]), (RM_taskset[i][1]/ RM_taskset[i][0]),  (RM_taskset[i][0]/ RM_taskset[i][0]),  RM_taskset[i][0] - RM_taskset[i][1],  RM_taskset[i][0] - RM_taskset[i][0],  RM_taskset[i][0] - RM_taskset[i][1]])
                attr = np.round(attr, 3)
                #attr = torch.tensor(attr).to(torch.float32).to(device)
                #print(attr)
                line1, line2 = [], []
                for i in range(numtasks):
                    for j in range(numtasks-i-1):
                        line1.append(i)
                for i in range(numtasks):
                    for j in range(i+1, numtasks):
                        line2.append(j)
                edge_index = [line1,line2]
                weights = procdata.make_edge_uniproc_T_C(RM_taskset, numtasks)
                #convert_for_y = torch.tensor(FBBFFD_sche).to(device)

                total_attr.append(attr)
                total_FBBFFD_sche.append(FBBFFD_sche)
                total_weights.append(weights.tolist())
                total_edges.append(edge_index)

                break


        total_attr = torch.tensor(total_attr).to(device)
        total_FBBFFD_sche = torch.tensor(total_FBBFFD_sche).to(device)
        total_edges = torch.tensor(total_edges).to(device)        
        total_weights = torch.tensor(total_weights).to(device)    

        return total_attr, total_FBBFFD_sche, total_edges, total_weights



    def gen_dataset_binomial(self, numtasks, num_processor, MAX_period, dataset_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generate_tasks = GenSet()
        procdata = Procdata()
        generate_mask = GenMask()
        heuristic_fbbffd = FBBFFD()
        rm_rta = RM_RTA()
        
        duplicate_check = []
        total_attr = []
        total_FBBFFD_sche = []
        total_edges = []
        total_weights = []

        for size in range(dataset_size):
            if size%500 == 0:
                print(size)
            while True:
                tasks_to_gen = random.choice(range(num_processor+1, numtasks))
                #taskset = generate_tasks.makeTC_binomial(tasks_to_gen, num_processor, MAX_period)
                taskset = generate_tasks.makeTC(tasks_to_gen, num_processor, MAX_period)
                # Rate Monotonic
                RM_taskset = generate_tasks.rate_monotonic(taskset)
                
                if size%3 == 0 :
                    ANS_heuristic, unschedulable_with_ANS = heuristic_fbbffd.rtffd(RM_taskset, num_processor)
                elif size%3 == 1:
                    ANS_heuristic, unschedulable_with_ANS = heuristic_fbbffd.bestfit(RM_taskset, num_processor)
                else:
                    ANS_heuristic, unschedulable_with_ANS = heuristic_fbbffd.worstfit(RM_taskset, num_processor)

                FBBFFD_heuristic_induced_proc, unschedulable_with_FBBFDD_induced_proc = heuristic_fbbffd.rtffd(RM_taskset, num_processor-1)
                
                Bestfit_induced_proc, unschedulable_with_Bestfit_induced_proc = heuristic_fbbffd.bestfit(RM_taskset, num_processor-1)
                
                Worstfit_induced_proc, unschedulable_with_Worstfit_induced_proc = heuristic_fbbffd.worstfit(RM_taskset, num_processor-1)

                for sublist in duplicate_check:
                    if RM_taskset == sublist:
                #        print(RM_taskset)
                        found_duplicate = True
                        continue
                        
                if unschedulable_with_ANS:
                    continue

                if not unschedulable_with_FBBFDD_induced_proc:
                    continue
                    
                if not unschedulable_with_Bestfit_induced_proc:
                    continue
                    
                if not unschedulable_with_Worstfit_induced_proc:
                    continue

                #print(tasks_to_gen)

                FBBFFD_sche = [999] * tasks_to_gen

                #print(RM_taskset)
                #print(FBBFFD_heuristic)
                #duplicate_check.append(RM_taskset)
                
                for i in range(len(RM_taskset)):
                    for j in range(len(ANS_heuristic)):
                        for x in ANS_heuristic[j]:
                            if RM_taskset[i] == x:
                                FBBFFD_sche[i] = j
                                break

                one_hot_FBBFFD_sche = []
                for i in range(tasks_to_gen):
                    list_sche = [0] * num_processor
                    list_sche[FBBFFD_sche[i]] = 1
                    one_hot_FBBFFD_sche.append(list_sche)

                attr = []
                for i in range(tasks_to_gen):
                    attr.append([RM_taskset[i][0], RM_taskset[i][1], 
                    (RM_taskset[i][0]/RM_taskset[i][1]),
                    RM_taskset[i][0] - RM_taskset[i][1]])
                attr = np.array(np.round(attr, 3), dtype=np.float32)
                #attr = torch.tensor(attr).to(torch.float32).to(device)
                #print(attr)
                line1, line2 = [], []
                for i in range(tasks_to_gen):
                    for j in range(tasks_to_gen-i-1):
                        line1.append(i)
                for i in range(tasks_to_gen):
                    for j in range(i+1, tasks_to_gen):
                        line2.append(j)
                edge_index = [line1,line2]
                weights = procdata.make_edge_uniproc_T_C(RM_taskset, numtasks, tasks_to_gen)
                #convert_for_y = torch.tensor(FBBFFD_sche).to(device)
                attr = torch.tensor(attr)
                one_hot_FBBFFD_sche = torch.tensor(one_hot_FBBFFD_sche, dtype=torch.float64)
                weights = torch.tensor(weights.tolist())
                edge_index = torch.tensor(edge_index)

                total_attr.append(attr)
                total_FBBFFD_sche.append(one_hot_FBBFFD_sche)
                total_weights.append(weights)
                total_edges.append(edge_index)

                break

        #total_attr = torch.tensor(total_attr).to(device)
        #total_FBBFFD_sche = torch.tensor(total_FBBFFD_sche).to(device)
        #total_edges = torch.tensor(total_edges).to(device)        
        #total_weights = torch.tensor(total_weights).to(device)    

        return total_attr, total_FBBFFD_sche, total_edges, total_weights
        
        
    def gen_dataset_testing(self, numtasks, num_processor, MAX_period, dataset_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generate_tasks = GenSet()
        procdata = Procdata()
        generate_mask = GenMask()
        heuristic_fbbffd = FBBFFD()
        rm_rta = RM_RTA()
        
        #duplicate_check = []
        total_attr = []
        total_FBBFFD_sche = []
        total_edges = []
        total_weights = []

        for size in range(dataset_size):
            if size%500 == 0:
                print(size)
            
            while True:
                new_task_C = 1
                tasks_to_gen = random.choice(range(num_processor+1, numtasks))
                taskset = generate_tasks.makeTC(tasks_to_gen, num_processor, MAX_period)
                # Rate Monotonic
                RM_taskset = generate_tasks.rate_monotonic(taskset)
                RM_taskset_with_new_task = list(map(list, RM_taskset))
                RM_taskset_with_new_task.append([MAX_period, 1])
                RM_taskset_with_new_task_900 = list(map(list, RM_taskset))
                RM_taskset_with_new_task_900.append([MAX_period, 900])
                
                RTFFD_heuristic, unschedulable_with_RTFFD = heuristic_fbbffd.rtffd(RM_taskset_with_new_task, num_processor)
                RTFFD_900_heuristic, unschedulable_with_RTFFD_900 = heuristic_fbbffd.rtffd(RM_taskset_with_new_task_900, num_processor)
                Bestfit_heuristic, unschedulable_with_Bestfit = heuristic_fbbffd.bestfit(RM_taskset_with_new_task, num_processor)
                Bestfit_900_heuristic, unschedulable_with_Bestfit_900 = heuristic_fbbffd.bestfit(RM_taskset_with_new_task_900, num_processor)
                Worstfit_heuristic, unschedulable_with_Worstfit = heuristic_fbbffd.worstfit(RM_taskset_with_new_task, num_processor)
                Worstfit_900_heuristic, unschedulable_with_Worstfit_900 = heuristic_fbbffd.worstfit(RM_taskset_with_new_task_900, num_processor)
                
                if unschedulable_with_RTFFD and unschedulable_with_Bestfit and unschedulable_with_Worstfit:
                    continue
                if not unschedulable_with_RTFFD_900:
                    continue
                if not unschedulable_with_Bestfit_900:
                    continue
                if not unschedulable_with_Worstfit_900:
                    continue
                
                else:
                    while True:
                        new_task_C += 1
                        RM_taskset_with_new_task = list(map(list, RM_taskset))
                        RM_taskset_with_new_task.append([MAX_period, new_task_C])

                        
                        FBBFFD_heuristic_induced_proc, unschedulable_with_FBBFDD_induced_proc = heuristic_fbbffd.rtffd(RM_taskset_with_new_task, num_processor)
                
                        Bestfit_induced_proc, unschedulable_with_Bestfit_induced_proc = heuristic_fbbffd.bestfit(RM_taskset_with_new_task, num_processor)
                
                        Worstfit_induced_proc, unschedulable_with_Worstfit_induced_proc = heuristic_fbbffd.worstfit(RM_taskset_with_new_task, num_processor)
                
                        if new_task_C == MAX_period:
                            break
                            
                        if unschedulable_with_FBBFDD_induced_proc and unschedulable_with_Bestfit_induced_proc and unschedulable_with_Worstfit_induced_proc:
                            break
                            
                    one_hot_FBBFFD_sche = []
                    for i in range(tasks_to_gen+1):
                        list_sche = [0] * num_processor
                        one_hot_FBBFFD_sche.append(list_sche)
                        
                    attr = []
                    for i in range(tasks_to_gen+1):
                        attr.append([RM_taskset_with_new_task[i][0], RM_taskset_with_new_task[i][1], (RM_taskset_with_new_task[i][0]/RM_taskset_with_new_task[i][1]), RM_taskset_with_new_task[i][0] - RM_taskset_with_new_task[i][1]])
                    attr = np.array(np.round(attr, 3), dtype=np.float32)
                    
                    line1, line2 = [], []
                    for i in range(tasks_to_gen+1):
                        for j in range(tasks_to_gen-i):
                            line1.append(i)
                    for i in range(tasks_to_gen+1):
                        for j in range(i+1, tasks_to_gen+1):
                            line2.append(j)
                            
                    edge_index = [line1,line2]
                    weights = procdata.make_edge_uniproc_T_C(RM_taskset_with_new_task, numtasks, tasks_to_gen+1)
                    one_hot_FBBFFD_sche = torch.tensor(one_hot_FBBFFD_sche, dtype=torch.float64)
                    attr = torch.tensor(attr)
                    weights = torch.tensor(weights.tolist())
                    edge_index = torch.tensor(edge_index)

                    total_attr.append(attr)
                    total_weights.append(weights)
                    total_FBBFFD_sche.append(one_hot_FBBFFD_sche)
                    total_edges.append(edge_index)

                    break 

        return total_attr, total_FBBFFD_sche, total_edges, total_weights
