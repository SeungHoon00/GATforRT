from dataclasses import dataclass 

class FBBFFD:

    def RBF(self, task, t):
        result = task[1] + (task[1]/task[0]) * t
        return result
        
    def Util(self, numtasks):
        return numtasks*(pow(2, 1/numtasks)-1.0)



    def fbbffd(self, task, numproc):

        bin_pack = [[] for i in range(numproc)]
        unschedulable = False
        cnt_assigned = 0
        for i in range(len(task)):
            for j in range(numproc):
                sum_RBF = 0
                sum_util = 0
                for in_numproc in range(len(bin_pack[j])):
                #RBF(Request Bound Function)
                    sum_RBF += self.RBF(bin_pack[j][in_numproc],task[i][0])
                    sum_util += (bin_pack[j][in_numproc][1]/bin_pack[j][in_numproc][0])
                if(((task[i][0] - sum_RBF) >= task[i][1]) and ((1-sum_util)>= (task[i][1]/task[i][0]))):
                    bin_pack[j].append(task[i])
                    cnt_assigned += 1
                    break
            
        #print(cnt_assigned)
            
        if cnt_assigned < len(task):
            unschedulable = True


        return bin_pack, unschedulable
        
        
        
    def rtffd(self, task, numproc):

        bin_pack = [[] for i in range(numproc)]
        unschedulable = False
        cnt_assigned = 0
        for i in range(len(task)):
            for j in range(numproc):
                sum_RBF = 0
                for in_numproc in range(len(bin_pack[j])):
                #RBF(Request Bound Function)
                    sum_RBF += self.RBF(bin_pack[j][in_numproc],task[i][0])
                if((task[i][1] + sum_RBF) <= task[i][0]):
                    bin_pack[j].append(task[i])
                    cnt_assigned += 1
                    break
            
        #print(cnt_assigned)
            
        if cnt_assigned < len(task):
            unschedulable = True


        return bin_pack, unschedulable
        
        
    def bestfit(self, task, numproc):

        bin_pack = [[] for i in range(numproc)]
        unschedulable = False
        cnt_assigned = 0
        for i in range(len(task)):
            sum_util = [1.0] * numproc
            for j in range(numproc):
                util = 0
                proc_bound = self.Util(len(bin_pack[j])+1)
                for in_numproc in range(len(bin_pack[j])):
                    util += (bin_pack[j][in_numproc][1]/bin_pack[j][in_numproc][0])
                sum_util[j] = proc_bound - util
            try:    
                bestfit_index = sum_util.index(min(k for k in sum_util if k - (task[i][1]/task[i][0]) >= 0))
                bin_pack[bestfit_index].append(task[i])
                cnt_assigned += 1
            except:
                unschedulable = True
                break
            
        #print(cnt_assigned)
            
        if cnt_assigned < len(task):
            unschedulable = True


        return bin_pack, unschedulable
        
    def worstfit(self, task, numproc):

        bin_pack = [[] for i in range(numproc)]
        unschedulable = False
        cnt_assigned = 0
        for i in range(len(task)):
            sum_util = [1.0] * numproc
            for j in range(numproc):
                util = 0
                proc_bound = self.Util(len(bin_pack[j])+1)
                for in_numproc in range(len(bin_pack[j])):
                    util += (bin_pack[j][in_numproc][1]/bin_pack[j][in_numproc][0])
                sum_util[j] = proc_bound - util
            try:    
                bestfit_index = sum_util.index(max(k for k in sum_util if k - (task[i][1]/task[i][0]) >= 0))
                bin_pack[bestfit_index].append(task[i])
                cnt_assigned += 1
            except:
                unschedulable = True
                break
        #print(cnt_assigned)
            
        if cnt_assigned < len(task):
            unschedulable = True


        return bin_pack, unschedulable
