#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>

using namespace std;

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------

vector< vector<double> > I;
vector<int> J, qJ, L, qL;
vector < vector<double> > DM;

//-----------------------------------------------------------------------------
// Class Solution
//-----------------------------------------------------------------------------

class Solution {
public:
    vector <int> locations;
    vector <double> objectives;
    int dominanceRank;
    
    Solution(int numLocations, int numObjectives) {
        locations.resize(numLocations);
        objectives.resize(numObjectives);
        dominanceRank = 0;
    }
    void print() {
        for (int i=0; i<locations.size(); i++) {
            cout << locations[i] << "\t";
        }
        for (int i=0; i<objectives.size(); i++) {
            cout << objectives[i] << "\t";
        }
        cout << dominanceRank << endl;
    }
};

//-----------------------------------------------------------------------------
// Load problem data
//-----------------------------------------------------------------------------

void loadData(string problemFile, string demandsFile) {
    int n;
    
    //----- Preexistings and candidates ----------------------------------------
    ifstream file(problemFile);
    string s;
    while (file >> s && s != "-----BEGIN-----");
    file  >> n;
    J.resize(n);
    qJ.resize(n);
    for (int i=0; i<n; i++) {
        file >> J[i];
        file >> qJ[i];
    }
    file >> n;
    L.resize(n);
    qL.resize(n);
    for (int i=0; i<n; i++) {
        file >> L[i];
        file >> qL[i];
    }
    file.close();
   
    //----- Demands -----------------------------------------------------------
    file.open(demandsFile);
    file >> n;
    vector<double> row;
    row.resize(3);
    for (int i=0; i<n; i++) {
        for (int j=0; j<3; j++) file >> row[j];
        I.push_back(row);
    }
    file.close();
}

//-----------------------------------------------------------------------------
// Distance between two geographical points
//-----------------------------------------------------------------------------

// Haversine distance
double HaversineDistance(double lat1, double lon1, double lat2, double lon2) {
	double dlon = fabs(lon1 - lon2);
	double dlat = fabs(lat1 - lat2);
	double aa = pow((sin((double)dlat/(double)2*0.01745)),2) + cos(lat1*0.01745) *
               cos(lat2*0.01745) * pow((sin((double)dlon/(double)2*0.01745)),2);
	double c = 2 * atan2(sqrt(aa), sqrt(1-aa));
	double d = 6371 * c;
	return round(d);
}

// Distance from Distance Matrix
double distance(int i, int j) {
	if (i >= j)	return DM[i][j];
	else return DM[j][i];
}

// Distance matrix
void makeDistanceMatrix() {
	DM.resize(I.size());
	for (int i=0; i<DM.size(); i++) {
		for (int j=0; j<=i; j++) {
			DM[i].push_back(HaversineDistance(I[i][0], I[i][1], I[j][0], I[j][1]));
		}
	}
}

//-----------------------------------------------------------------------------
// Customer behavior: Binary
//-----------------------------------------------------------------------------

double utilityBinary(vector<int>& X) {
   
   double bestX, bestJ, attr;
   
   double total = 0;
   double utility = 0;
   for (int i=0; i<I.size(); i++) {
      total += I[i][2];
      bestJ = -1;
      for (int j=0; j<J.size(); j++) {
         attr = (double)qJ[j]/(1+distance(i, J[j]));
         if (attr > bestJ) bestJ = attr;              
      }
      bestX = -1;
      for (int j=0; j<X.size(); j++) {
         attr = (double)qL[X[j]]/(1+distance(i, L[X[j]]));
         if (attr > bestX) bestX = attr;
      }

      if (bestX > bestJ) utility += I[i][2];
      else if (bestX == bestJ) utility +=  I[i][2]/2.0;
   }
   return utility / total * 100;
}

//-----------------------------------------------------------------------------
// Customer behavior: Proportional
//-----------------------------------------------------------------------------

double utilityProportional(vector<int>& X) {
   
   double attrJ, attrX;
   
   double total = 0;
   double utility = 0;
   for (int i=0; i<I.size(); i++) {
      total += I[i][2];
      
      attrJ = 0;
      for (int j=0; j<J.size(); j++) {
         attrJ += (double)qJ[j]/(1+distance(i, J[j]));
      }
      
      attrX = 0;
      for (int j=0; j<X.size(); j++) {
         attrX += (double)qL[X[j]]/(1+distance(i, L[X[j]]));
      }

      utility += I[i][2] * (attrX) / (attrJ + attrX);
   }
   return utility / total * 100;
}
