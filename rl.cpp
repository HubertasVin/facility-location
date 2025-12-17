#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "dcflp.cpp"
#include "globals.cpp"

using namespace std;

unordered_map<string, double> Q; // Q-Table

bool performTraining = true; // If true, train; else just pick greedily (but your picker still uses epsilon)

//===== Chart data file streams ===============================================
mutex file_mutex;
ofstream trainingCurvesFile;
ofstream finalScoresFile;

// mark how an action was chosen (set inside getActionEpsilonGreedy)
static bool g_last_pick_was_random = false;

//===== Function prototypes ===================================================
string stateActionToString(const vector<int> &state, int action);
string stateToString(const vector<int> &state);
string stateToStringCanonical(vector<int> state);
int getActionEpsilonGreedyLocal(const vector<int> &state, double epsilon1,
                                const unordered_map<string, double> &Q_local);
int getActionGreedyLocal(const vector<int> &state, const unordered_map<string, double> &Q_local);
void updateQTable(const string &stateActionStr, const string &nextStateStr, const vector<int> &nextState,
                  double reward, unordered_map<string, double> &Q_local);
void saveQTable(const unordered_map<string, double> &Q, const string &filename);
void loadQTable(unordered_map<string, double> &Q, const string &filename);
void trainQAgent(int episodes, int maxFacilities, int run_id, unordered_map<string, double> &Q_local);
void runMultipleTrainingRuns(int numRuns, int episodes, int maxFacilities, const string &qTableFile);
vector<int> getOptimalSolutionLocal(int maxFacilities, unordered_map<string, double> Q_local);

//===== NEW: diagnostics grounded in utilityBinary ============================

// Thin wrapper: call utilityBinary with a **mutable** copy (DCFLP expects non-const ref).
static double U(const vector<int> &s_const)
{
    vector<int> s = s_const;
    sort(s.begin(), s.end()); // be safe: many utilities assume sorted
    return utilityBinary(s);  // DCFLP.cpp: double utilityBinary(vector<int>&)
}

// Build list of available actions (not already in state)
static vector<int> availableActions(const vector<int> &state)
{
    vector<int> avail;
    avail.reserve(L.size());
    for (int loc : L)
        if (find(state.begin(), state.end(), loc) == state.end())
            avail.push_back(loc);
    return avail;
}

// For a state, compute true marginal gain r(a) = U(s∪{a}) - U(s)
static vector<pair<int, double>> marginalGains(const vector<int> &state)
{
    double u_before = U(state);
    vector<pair<int, double>> gains;
    for (int a : availableActions(state))
    {
        vector<int> next = state;
        next.push_back(a);
        sort(next.begin(), next.end());
        double u_after = U(next);
        gains.emplace_back(a, u_after - u_before);
    }
    sort(gains.begin(), gains.end(), [](auto &x, auto &y)
         { return x.second > y.second; });
    return gains;
}

// Q-table access for explanations
static double getQ(const vector<int> &s, int a)
{
    string k = stateActionToString(s, a);
    auto it = Q.find(k);
    return (it == Q.end()) ? 0.0 : it->second;
}
static double maxFutureQ_existingKeys(const vector<int> &sNext)
{
    double best = -1e300;
    for (int a : availableActions(sNext))
    {
        double q = getQ(sNext, a);
        if (q > best)
            best = q;
    }
    if (best == -1e300)
        best = 0.0;
    return best;
}

//===== Main ==================================================================
int main()
{
    srand(static_cast<unsigned int>(time(0))); // Seed for random numbers

    loadData(problemFile, demandsFile); // Load problem data
    makeDistanceMatrix();               // Make distance matrix
    // loadQTable(Q, qTableFile);          // Load Q-Table

    if (performTraining)
    {
        runMultipleTrainingRuns(num_training_runs, trainingEpisodes, maxFacilities, qTableFile);
        // saveQTable(Q, qTableFile);
    }

    // Generate a solution (note: still uses epsilon-greedy per your code)
    vector<int> optimalSolution = getOptimalSolutionLocal(maxFacilities, Q);
    cout << "Optimal locations for the new facilities: ";
    for (int loc : optimalSolution)
        cout << loc << " ";
    cout << "(" << U(optimalSolution) << "%)" << endl;

    return 0;
}

//=============================================================================
//===== Q-learning and helpers =================================================
//=============================================================================

double epsilonValue(int episode) { return 1 / (1 + episode / epsilon_slope); }

string stateActionToString(const vector<int> &state, int action)
{
    string str;
    for (int loc : state)
        str += to_string(loc) + "-";
    str += "|" + to_string(action);
    return str;
}

string stateToString(const vector<int> &state)
{
    string str;
    for (int loc : state)
        str += to_string(loc) + "-";
    return str;
}

string stateToStringCanonical(vector<int> state)
{
    sort(state.begin(), state.end());
    string str;
    for (int loc : state)
        str += to_string(loc) + "-";
    return str;
}

int getActionEpsilonGreedyLocal(const vector<int> &state, double epsilon1,
                                const unordered_map<string, double> &Q_local)
{
    double randomValue = static_cast<double>(rand()) / RAND_MAX;
    if (randomValue < epsilon1)
    {
        int randomFacility;
        do
        {
            randomFacility = L[rand() % L.size()];
        } while (find(state.begin(), state.end(), randomFacility) != state.end());
        g_last_pick_was_random = true;
        return randomFacility;
    }
    int bestFacility = getActionGreedyLocal(state, Q_local);
    g_last_pick_was_random = false;
    return bestFacility;
}

int getActionGreedyLocal(const vector<int> &state, const unordered_map<string, double> &Q_local)
{
    double maxQValue = -1e9;
    int bestFacility = -1;
    for (int loc : L)
    {
        if (find(state.begin(), state.end(), loc) == state.end())
        {
            string stateActionStr = stateActionToString(state, loc);
            auto it = Q_local.find(stateActionStr);
            if (it != Q_local.end() && it->second > maxQValue)
            {
                maxQValue = it->second;
                bestFacility = loc;
            }
        }
    }
    return (bestFacility == -1) ? L[rand() % L.size()] : bestFacility;
}

void updateQTable(const string &stateActionStr, const string &nextStateStr, const vector<int> &nextState,
                  double reward, unordered_map<string, double> &Q_local)
{
    double qPredict = Q_local[stateActionStr];
    double maxFutureReward = -1e9;
    for (int loc : L)
    {
        if (find(nextState.begin(), nextState.end(), loc) == nextState.end())
        {
            string nextStateActionStr = stateActionToString(nextState, loc);
            auto it = Q_local.find(nextStateActionStr);
            if (it != Q_local.end())
                maxFutureReward = max(maxFutureReward, it->second);
        }
    }
    if (maxFutureReward == -1e9)
        maxFutureReward = 0;
    double qTarget = reward + gamma_rl * maxFutureReward;
    Q_local[stateActionStr] += alpha * (qTarget - qPredict);
}

void loadQTable(unordered_map<string, double> &Q, const string &filename)
{
    ifstream inFile(filename);
    if (inFile.is_open())
    {
        string line;
        while (getline(inFile, line))
        {
            stringstream ss(line);
            string key;
            double value;
            ss >> key >> value;
            Q[key] = value;
        }
        inFile.close();
        cout << "Q-Table loaded from " << filename << endl;
    }
    else
    {
        cerr << "Unable to open file for loading Q-Table." << endl;
    }
}

void saveQTable(const unordered_map<string, double> &Q, const string &filename)
{
    ofstream outFile(filename);
    if (outFile.is_open())
    {
        for (const auto &entry : Q)
            outFile << entry.first << " " << entry.second << "\n";
        outFile.close();
        cout << "Q-Table saved to " << filename << endl;
    }
    else
    {
        cerr << "Unable to open file for saving Q-Table." << endl;
    }
}

void trainQAgent(int episodes, int maxFacilities, int run_id, unordered_map<string, double> &Q_local)
{
    double utility_sum = 0.0;
    int window_count = 0;

    for (int episode = 0; episode < episodes; ++episode)
    {
        auto start_time = chrono::high_resolution_clock::now();
        vector<int> currentState;
        double epsilon_current = fixed_epsilon ? epsilon : epsilonValue(episode);

        if (useFinalRewardQ)
        {
            // ============================================================
            // NEW METHOD: Q-Learning with final reward as the reward
            // ============================================================

            vector<tuple<string, string, vector<int>>> trajectory;

            for (int step = 0; step < maxFacilities; ++step)
            {
                int newFacility = getActionEpsilonGreedyLocal(currentState, epsilon_current, Q_local);

                vector<int> nextState = currentState;
                nextState.push_back(newFacility);
                sort(nextState.begin(), nextState.end());

                string stateActionStr = stateActionToString(currentState, newFacility);
                string nextStateStr = stateToString(nextState);

                trajectory.emplace_back(stateActionStr, nextStateStr, nextState);

                currentState = nextState;
            }

            double finalUtility = U(currentState);

            for (const auto &[stateActionStr, nextStateStr, nextState] : trajectory)
            {
                double qPredict = Q_local[stateActionStr];

                double maxFutureReward = -1e9;
                for (int loc : L)
                {
                    if (find(nextState.begin(), nextState.end(), loc) == nextState.end())
                    {
                        string nextStateActionStr = stateActionToString(nextState, loc);
                        auto it = Q_local.find(nextStateActionStr);
                        if (it != Q_local.end())
                            maxFutureReward = max(maxFutureReward, it->second);
                    }
                }
                if (maxFutureReward == -1e9)
                    maxFutureReward = 0;

                // Q-Learning update with final utility as reward
                double qTarget = finalUtility + gamma_rl * maxFutureReward;
                Q_local[stateActionStr] += alpha * (qTarget - qPredict);
            }
        }
        else
        {
            // ============================================================
            // Q-LEARNING: intermediate reward
            // ============================================================

            for (int step = 0; step < maxFacilities; ++step)
            {
                int newFacility = getActionEpsilonGreedyLocal(currentState, epsilon_current, Q_local);

                vector<int> nextState = currentState;
                nextState.push_back(newFacility);
                sort(nextState.begin(), nextState.end());

                double currentUtility = U(currentState);
                double nextUtility = U(nextState);
                double reward = nextUtility - currentUtility;

                string stateActionStr = stateActionToString(currentState, newFacility);
                string nextStateStr = stateToString(nextState);

                updateQTable(stateActionStr, nextStateStr, nextState, reward, Q_local);

                currentState = nextState;
            }
        }

        double final_utility = U(currentState);
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;

        utility_sum += final_utility;
        window_count++;

        if (window_count == window_size)
        {
            vector<int> greedy_solution = getOptimalSolutionLocal(maxFacilities, Q_local);
            double greedy_utility = U(greedy_solution);

            {
                std::lock_guard<std::mutex> lock(file_mutex);
                trainingCurvesFile << run_id << "," << episode << "," << fixed << setprecision(6) << greedy_utility << ","
                                   << elapsed.count() << "," << epsilon_current << endl;
            }

            utility_sum = 0.0;
            window_count = 0;
        }
    }
}

void runMultipleTrainingRuns(int numRuns, int episodes, int maxFacilities, const string &qTableFile)
{
    trainingCurvesFile.open("training_curves.csv");
    trainingCurvesFile << "run_id,episode,utility,runtime_seconds,epsilon\n";

    finalScoresFile.open("final_scores.csv");
    finalScoresFile << "run_id,final_utility,total_runtime,num_facilities_selected\n";

    cout << "\n=== Training Run: " << endl;
#pragma omp parallel for schedule(dynamic)
    for (int run = 0; run < numRuns; ++run)
    {
        unordered_map<string, double> Q_local;

#pragma omp critical
        {
            cout << run + 1 << " (t" << omp_get_thread_num() << ")" << endl;
        }

        auto run_start = chrono::high_resolution_clock::now();
        trainQAgent(episodes, maxFacilities, run, Q_local);
        vector<int> solution = getOptimalSolutionLocal(maxFacilities, Q_local);
        double final_utility = U(solution);
        auto run_end = chrono::high_resolution_clock::now();
        chrono::duration<double> total_time = run_end - run_start;

        {
            std::lock_guard<std::mutex> lock(file_mutex);
            finalScoresFile << run << "," << fixed << setprecision(6) << final_utility << "," << total_time.count()
                            << "," << solution.size() << "\n";
        }

        saveQTable(Q_local, qTableFile.substr(0, qTableFile.find_last_of('.')) + "_run_" + to_string(run) + ".dat");
    }

    cout << " ===\n";

    trainingCurvesFile.close();
    finalScoresFile.close();
}

vector<int> getOptimalSolutionLocal(int maxfacilities, unordered_map<string, double> Q_local)
{
    vector<int> currentstate;
    for (int step = 0; step < maxfacilities; ++step)
    {
        int bestfacility = getActionGreedyLocal(currentstate, Q_local);
        currentstate.push_back(bestfacility);
        sort(currentstate.begin(), currentstate.end());
    }
    return currentstate;
}
