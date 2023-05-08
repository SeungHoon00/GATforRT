import torch

class GenMask:

    def makeMask(self, numtasks):
        
        num = int(numtasks/3)
        num_for_train = numtasks-(num*2)

        Train_Mask, Val_Mask, Test_Mask = torch.zeros(numtasks, dtype=torch.bool), torch.zeros(numtasks, dtype=torch.bool), torch.zeros(numtasks, dtype=torch.bool)
        for i in range(num_for_train):
            Train_Mask[i] = True
            Val_Mask[i] = False
            Test_Mask[i] = False


        for i in range(num_for_train, num_for_train+num):
            Train_Mask[i] = False
            Val_Mask[i] = True
            Test_Mask[i] = False


        for i in range(numtasks-num, numtasks):
            Train_Mask[i] = False
            Val_Mask[i] = False
            Test_Mask[i] = True

        return Train_Mask, Val_Mask, Test_Mask
