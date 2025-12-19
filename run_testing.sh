#!/bin/bash

alphas=(0.1 0.2 0.4) # 0.8 removed because value already tested in variable epsilon
gammas=(0 0.4 0.5)
epsilons=(0.05 0.1 0.2 0.3)

export RL_QTABLE_FILE="../qtables/qtable.dat"
export RL_DEMANDS_FILE="../demands_1000LT.dat"
export RL_PROBLEM_FILE="../CFLP_J10L100.dat"
export RL_MAX_FACILITIES=5

mkdir -p results
cd results
make -C .. build

echo "Starting automated training runs..."
echo "Total configurations: $((${#alphas[@]} * ${#gammas[@]} * ${#epsilons[@]}))"


run_with_parmeters() {
    alpha=$1
    gamma=$2
    epsilon=$3

    echo ""
    echo "=========================================="
    echo "Alpha: $alpha, Gamma: $gamma, Epsilon: $epsilon"
    echo "=========================================="

    # Set environment variables
    export RL_ALPHA=$alpha
    export RL_GAMMA=$gamma
    export RL_EPSILON=$epsilon
    export RL_NUM_RUNS=40
    export RL_WINDOW_SIZE=5

    ../rl

    if [ -f "training_curves.csv" ]; then
        mv training_curves.csv "data/training_a${alpha}_g${gamma}_e${epsilon}.csv"
        mv final_scores.csv "data/final_a${alpha}_g${gamma}_e${epsilon}.csv"
    else
        echo "ERROR: training_curves.csv not found!"
    fi
}

export RL_EPISODES=40000
export RL_FIXED_EPSILON=true
for alpha in "${alphas[@]}"; do
	run_with_parmeters $alpha 0.2 0.2
done
export RL_EPISODES=20000
for gamma in "${gammas[@]}"; do
	run_with_parmeters 0.8 $gamma 0.2
done
for epsilon in "${epsilons[@]}"; do
	run_with_parmeters 0.8 0.2 $epsilon
done
export RL_FIXED_EPSILON=false
run_with_parmeters 0.8 0.2 0

python3 ../viz_hyperparams.py --treat-e0-as-variable --dir data --outdir .




# Test if updating Q-values with final reward to makes a difference
export RL_ALPHA=0.4
export RL_GAMMA=0.4
export RL_EPSILON=0.3
export RL_FIXED_EPSILON=true
export RL_FINAL_REWARD_Q=0
export RL_EPISODES=30000
export RL_NUM_RUNS=12
export RL_WINDOW_SIZE=5
export RL_DEMANDS_FILE="../demands_1000LT.dat"
export RL_PROBLEM_FILE="../CFLP_J10L100.dat"
../rl
if [ -f "training_curves.csv" ]; then
    mv training_curves.csv "data/training_a${RL_ALPHA}_g${RL_GAMMA}_e${RL_EPSILON}_finalRewardOFF.csv"
    mv final_scores.csv "data/final_a${RL_ALPHA}_g${RL_GAMMA}_e${RL_EPSILON}_finalRewardOFF.csv"
else
    echo "ERROR: training_curves.csv not found!"
fi

export RL_FINAL_REWARD_Q=1
../rl
if [ -f "training_curves.csv" ]; then
    mv training_curves.csv "data/training_a${RL_ALPHA}_g${RL_GAMMA}_e${RL_EPSILON}_finalRewardON.csv"
    mv final_scores.csv "data/final_a${RL_ALPHA}_g${RL_GAMMA}_e${RL_EPSILON}_finalRewardON.csv"
else
    echo "ERROR: training_curves.csv not found!"
fi

python3 ../viz_finalreward.py --dir data --outdir .

echo "Done! Results saved in ./results/"
