import argparse
import numpy as np
import random
import pickle
import math
import torch
import os.path

def gen_dense(rows, cols):
    return np.random.normal(loc=0.0, scale = 0.25, size=(rows,cols)) 

def gen_sparse(rows, cols, col_nz):

    print(rows,cols)
    m = np.zeros((rows,cols),dtype="float32")
    if (rows % col_nz != 0):
        print("WARNING: rows is not evenly divided by col_nz")

    segments = col_nz
    seg_len = int(rows / col_nz)

    print("segs: {}, len: {}".format(segments, seg_len))

    for col in range(cols):
        for segment in range(segments):
            base_idx = seg_len * segment
            idx_in_seg = random.randint(0,seg_len-1)
            value = 1.0 / math.sqrt(segments)
            value = value * random.choice([-1.0, 1.0])

            m[base_idx + idx_in_seg, col] = value

    return m


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random Projection Matrix \
            Builder")
    parser.add_argument('rows', type=int)
    parser.add_argument('cols', type=int)
    parser.add_argument('cnz', type=int)
    parser.add_argument('mode', type=str)
    parser.add_argument('-o', type=str)

    args = parser.parse_args()

    q = []
    for i in range(26):
        if (args.mode == "sparse"):
            q.append(np.transpose(gen_sparse(args.rows,args.cols,args.cnz)))
        elif (args.mode == "dense"):
            q.append(np.transpose(gen_dense(args.rows, args.cols)))

    qq = np.array(q,dtype="float32")

    print(np.shape(qq))

    if os.path.exists(args.o):
        print("File exists")
    else:
        pickle.dump(qq, open(args.o, 'wb'))

