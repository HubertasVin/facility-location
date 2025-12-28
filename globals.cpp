#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

string getEnvVar(const char *name, const string &default_value)
{
    const char *val = getenv(name);
    return val ? string(val) : default_value;
}

int getEnvInt(const char *name, int default_value)
{
    const char *val = getenv(name);
    return val ? stoi(string(val)) : default_value;
}

double getEnvDouble(const char *name, double default_value)
{
    const char *val = getenv(name);
    return val ? stod(string(val)) : default_value;
}

bool getEnvBool(const char *name, bool default_value)
{
    const char *val = getenv(name);
    if (!val)
        return default_value;
    string s(val);
    return (s == "true" || s == "1" || s == "TRUE");
}

//===== Parameters for Facility location problem ==============================
string problemFile = getEnvVar("RL_PROBLEM_FILE", "CFLP.dat");   // Problem description file
string demandsFile = getEnvVar("RL_DEMANDS_FILE", "demands.dat"); // Demand points file
int maxFacilities = getEnvInt("RL_MAX_FACILITIES", 3);                   // Number of new facilities

//===== Parameters for Q-Learning =============================================
int trainingEpisodes = getEnvInt("RL_EPISODES", 20000);        // Training episodes
double alpha = getEnvDouble("RL_ALPHA", 0.8);                  // Learning rate
double gamma_rl = getEnvDouble("RL_GAMMA", 0.0);               // Discount factor
bool fixed_epsilon = getEnvBool("RL_FIXED_EPSILON", true);     // Use fixed epsilon value if true, slope function if false
double epsilon = getEnvDouble("RL_EPSILON", 0.2);              // Exploration rate
string qTableFile = getEnvVar("RL_QTABLE_FILE", "qtable.dat"); // Q-Table file
int num_training_runs = getEnvInt("RL_NUM_RUNS", 1);           // Number of runs to perform
int window_size = getEnvInt("RL_WINDOW_SIZE", 1);              // Window size for averaging training output
double epsilon_slope = ((double)trainingEpisodes / 2) /
                       9;                                      // Epsilon decreasing slope steepness value ((num. of episodes to reach 0.1 value) / 9)
bool useFinalRewardQ = getEnvBool("RL_FINAL_REWARD_Q", false); // Use final reward to update the Q-values if true
