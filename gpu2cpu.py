import sys
import torch
import tqdm
import pickle

infile = sys.argv[1]
outfile = sys.argv[2]

print("loading pickle")
pkldata = pickle.load(open(infile, 'rb'))

for key in tqdm.tqdm(pkldata):
    pkldata[key]['embed'] = pkldata[key]['embed'].cpu()

print("saving pickle")
pickle.dump(pkldata, open(outfile, 'wb'))