#!/bin/bash

alphas=(0.05 0.1 0.5 0.8)
gammas=(0 0.4 0.8)
epsilons=(0.05 0.1 0.2 0.3)

mkdir -p results
cd results

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
	export RL_EPISODES=200000
	export RL_QTABLE_FILE="../qtable.dat"
	export RL_DEMANDS_FILE="../demands_1000LT.dat"
	export RL_PROBLEM_FILE="../CFLP_J10L100.dat"

    ../rl

    if [ -f "training_curves.csv" ]; then
        mv training_curves.csv "training_a${alpha}_g${gamma}_e${epsilon}.csv"
        mv final_scores.csv "final_a${alpha}_g${gamma}_e${epsilon}.csv"
    else
        echo "ERROR: training_curves.csv not found!"
    fi
}

export RL_FIXED_EPSILON=true
for alpha in "${alphas[@]}"; do
	run_with_parmeters $alpha 0.2 0.2
done
for gamma in "${gammas[@]}"; do
	run_with_parmeters 0.8 $gamma 0.2
done
for epsilon in "${epsilons[@]}"; do
	run_with_parmeters 0.8 0.2 $epsilon
done
export RL_FIXED_EPSILON=false
run_with_parmeters 0.8 0.2 0
# python3 ../analyze_metrics.py 0.8 0.2 0

python3 ../compare_all_configs.py

echo "Done! Results saved in ./results/"
