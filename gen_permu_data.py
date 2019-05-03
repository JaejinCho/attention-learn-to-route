import pickle
import random
import sys

tsp_conf = sys.argv[1] # 20, 50 or 100

fname='data/tsp/tsp{}_test_seed1234.pkl'.format(tsp_conf)
data = pickle.load(open(fname,'rb'))

for sample in data:
    random.shuffle(sample)

fout = 'data/tsp/tsp{}_test_seed1234_perm1.pkl'.format(tsp_conf)
pickle.dump(data,open(fout,'wb'))

for sample in data:
    random.shuffle(sample)

fout = 'data/tsp/tsp{}_test_seed1234_perm2.pkl'.format(tsp_conf)
pickle.dump(data,open(fout,'wb'))

