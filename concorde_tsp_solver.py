import sys
import pickle
import numpy as np
import scipy.stats
import time
from concorde.tsp import TSPSolver
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def tour_length(xs, ys, tour):
    '''
    tour: a list of indices
    '''
    coord_sorted = [np.array((xs[i], ys[i])) for i in tour]
    coord_sorted.append(np.array((xs[tour[0]],ys[tour[0]])))
    length = 0
    for i in range(len(coord_sorted)-1):
        length += np.linalg.norm(coord_sorted[i]-coord_sorted[i+1])
    return round(length,6)

def isopt(opt_len, pred_len, threshold=1e-6):
    return (abs(opt_len - pred_len) < threshold)

def save_data(data, fname):
    pickle.dump(data,open(fname,'wb'))

def plot_and_save(list_timing, list_isopt, out_file):
    plt.close()
    opt_data = np.array(list_timing)[np.array(list_isopt) == True]
    density = gaussian_kde(opt_data)
    xs = np.linspace(0,0.2,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs))
    #plt.show()
    nonopt_data = np.array(list_timing)[np.array(list_isopt) == False]
    density = gaussian_kde(nonopt_data)
    xs = np.linspace(0,0.2,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs))

    plt.savefig(out_file)

# read file
tsp_config = sys.argv[1] # 20, 50, or 100
fname_data = './data/tsp/tsp{}_test_seed1234.pkl'.format(tsp_config) # ADAPT
fname_pred = './results/tsp/tsp{}_test_seed1234/tsp{}_test_seed1234-pretrained_tsp_{}-greedy-t1-0-10000_ConfMeasure.pkl'.format(tsp_config, tsp_config, tsp_config)

data = pickle.load(open(fname_data,'rb'))
result = pickle.load(open(fname_pred,'rb'))
list_pred_len = list(map(lambda x: x[0],result[0]))
list_pred_conf = list(map(lambda x: x[-1],result[0]))

# solve TSP sample by sample and compare between optimal and predicted solutions
list_timing = []
list_isopt = []
list_opt_len = []
threshold=1e-6

for ix, sample in enumerate(data):
    # convert data format
    xs = list(map(lambda x: x[0], sample))
    ys = list(map(lambda x: x[1], sample))
    norm="EUC_2D"

    # run TSP solver
    solver = TSPSolver.from_data(list(map(lambda x: x*1e8, xs)),list(map(lambda x: x*1e8, ys)),norm = norm)
    start = time.time()
    tour_data = solver.solve()
    duration = time.time() - start
    opt_len = tour_length(xs, ys, tour_data.tour)

    list_opt_len.append(opt_len)
    list_timing.append(duration)
    list_isopt.append(isopt(opt_len, list_pred_len[ix],threshold))
# save
data_dict = {'list_opt_len':list_opt_len, 'list_timing':list_timing, 'list_isopt':list_isopt}
save_data(data_dict,'tsp{}_greedy_4correlation.pkl'.format(tsp_config)) # ADAPT

# results and plots
#results
print("Percentage of optimal solution: {}\n".format(np.sum(np.array(list_isopt) == True)/len(list_isopt)))
print("Time taken: \nOpt. sol: {}\nNon-opt. sol: {}\n".format(np.mean(np.array(list_timing)[np.array(list_isopt) == True]),np.mean(np.array(list_timing)[np.array(list_isopt) == False])))
print("Relationship between Optimalness & Searching runtime: \nSpearman correlation: {}\nPearson correlation: {}\n".format(scipy.stats.spearmanr(list_isopt,list_timing),scipy.stats.pearsonr(list_isopt,list_timing)))
print("Relationship between Confidence (log p) & Searching runtime: \nSpearman correlation: {}\nPearson correlation: {}\n".format(scipy.stats.spearmanr(list_pred_conf,list_timing),scipy.stats.pearsonr(list_pred_conf,list_timing)))

plot_and_save(list_timing, list_isopt, 'tsp{}_timeNopt_{}.png'.format(tsp_config, threshold)) # ADAPT
