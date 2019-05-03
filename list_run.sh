# This is command lines that run the experiments in the final report (for RL
# section). Though I trained a few models with modifications but here,
# pretrained models will be used (For the jobs possibly needed gpu, I specified
# as "# may gpu needed" and CUDA_VISIBLE_DEVICES=`free-gpu` needs to be set in
# the front of the command


## 3.1 Run-time versus Confidence measure

### 1. Generate the data
### $ python generate_data.py --problem all --name validation --seed 4321
### $ python generate_data.py --problem all --name test --seed 1234

### 2. Run greedy search to get confidence measure (log probability per sample)
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy # may gpu needed. TSP 50
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy # may gpu needed. TSP 100

### 3. Install pyconcorde
### $ git clone https://github.com/jvkersch/pyconcorde
### $ cd pyconcorde
### $ pip install -e .
### $ cd ..

### 4. Calculate the correlation and save graphs
### $ python concorde_tsp_solver.py 50
### $ python concorde_tsp_solver.py 100

### 5. Look at final outputs for correlation values and *.png files for the graphs in the final report



## 3.2 Permutation

### 1. Generate permutated data
### python gen_permu_data.py 50 # TSP 50
### python gen_permu_data.py 100 # TSP 100

### 2. Run greedy decoding
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp50_test_seed1234_perm1.pkl --model pretrained/tsp_50 --decode_strategy greedy # may gpu needed. TSP 50
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp50_test_seed1234_perm2.pkl --model pretrained/tsp_50 --decode_strategy greedy # may gpu needed. TSP 50
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp100_test_seed1234_perm1.pkl --model pretrained/tsp_100 --decode_strategy greedy # may gpu needed. TSP 100
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp100_test_seed1234_perm2.pkl --model pretrained/tsp_100 --decode_strategy greedy # may gpu needed. TSP 100

### 3. Observe the results are same regardless of input permutation



## 3.3 Beam search

### 1. Run beam search decoding width 20 and 50
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy bs --width 20 # may gpu needed. TSP 50
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy bs --width 50 # may gpu needed. TSP 50
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy bs --width 20 # may gpu needed. TSP 100
### $ CUDA_VISIBLE_DEVICES=`free-gpu` python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy bs --width 50 # may gpu needed. TSP 100

### 2. Compared the predicted length and runtime to greedy, and to sample with its width 1280 (Refer to the results in the paper for greedy & sample since run results are almost same as reported)
