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
        attr, fbbffd_sche, edge_index, weights =  preprocess.gen_dataset_binomial(30, 8, 1000, 500000)

        x, y= attr, fbbffd_sche
        a = weights

        with open('./generated_sets/attr_mixed_paper_fbbffd_8proc_500000.pkl', 'wb') as fp:
            pickle.dump(attr, fp)
        with open('./generated_sets/fbbffd_sche_mixed_paper_fbbffd_8proc_500000.pkl', 'wb') as fp:
            pickle.dump(fbbffd_sche, fp)
        with open('./generated_sets/edge_index_mixed_paper_fbbffd_8proc_500000.pkl', 'wb') as fp:
            pickle.dump(edge_index, fp)
        with open('./generated_sets/weights_mixed_paper_fbbffd_8proc_500000.pkl', 'wb') as fp:
            pickle.dump(weights, fp)
        with open('./generated_sets/attr_mixed_paper_fbbffd_8proc_500000.txt', 'a') as fp:
            fp.write(str(attr))


if __name__ == '__main__':
    Write().write_data()

