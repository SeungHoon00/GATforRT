import numpy as np
import torch
#from trainer import Trainer
from preprocess import Preprocessing
import pickle
import gzip


class Write:
    def write_data(self):
        preprocess = Preprocessing()
        # numtasks, num_processor, MAX_period, dataset_size
        attr, fbbffd_sche, edge_index, weights =  preprocess.gen_dataset_testing(48, 16, 1000, 10000)


        with open('./generated_sets/attr_testing_mixed_16proc_10000.pkl', 'wb') as fp:
            pickle.dump(attr, fp)
        with open('./generated_sets/fbbffd_sche_testing_mixed_16proc_10000.pkl', 'wb') as fp:
            pickle.dump(fbbffd_sche, fp)
        with open('./generated_sets/edge_testing_mixed_16proc_10000.pkl', 'wb') as fp:
            pickle.dump(edge_index, fp)
        with open('./generated_sets/weights_testing_mixed_16proc_10000.pkl', 'wb') as fp:
            pickle.dump(weights, fp)
        with open('./generated_sets/weights_testing_mixed_16proc_10000.txt', 'a') as fp:
            fp.write(str(weights))


if __name__ == '__main__':
    Write().write_data()

