import math

class RM_RTA:

    def rmrta(self, tasks, taskorder, num_processor):

        bin_pack = [[] for i in range(num_processor)]

        for i in range(len(tasks)):
            bin_pack[taskorder[i]].append(tasks[i])

        unschedulable = False
        #print(bin_pack)
        for i in range(num_processor):
            a, a_past = 0, -1

            if not (bin_pack[i]):
                continue
            while a != a_past:
                cal_a = 0
                if a == 0:
                    for j in range(len(bin_pack[i])):
                        a += bin_pack[i][j][1]
                cal_a = bin_pack[i][len(bin_pack[i])-1][1]

                for j in range(len(bin_pack[i])-1):
                    cal_a += math.ceil(a/bin_pack[i][j][0])*bin_pack[i][j][1]
                a_past = a
                a = cal_a
                if (a > bin_pack[i][len(bin_pack[i])-1][0]):
                    unschedulable = True
                    break

        return(unschedulable)
