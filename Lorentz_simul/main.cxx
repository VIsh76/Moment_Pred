#include <iostream>
#include "lorenz95.h"
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

int main()
{
//  arma_rng::set_seed_random();
  int seed;
  cout << "Enter Seed" <<endl;
  cin>> seed;
  arma_rng::set_seed(seed);
  int dt;
  cout << "Enter dt" <<endl;
  cin >> dt;
  string model = "L63";
  cout << "Enter Sample Size" <<endl;
  int Nsample;
  cin >> Nsample;
  cout << "Enter Number of Simulation" <<endl;
  int Nsimul;
  cin >> Nsimul;
  cube F = zeros(Nsimul, 3, Nsample);
  mat X = 4*ones(3,Nsample)+randn(3,Nsample);
  for (int i=0; i<Nsimul; i=i+1){
    F.subcube(i, 0, 0, i , 2, Nsample-1)=X;
    M(X,dt,model);
  }
//  cout << "FileName :" <<endl;
//  cin>>filename;
  F.save(to_string(Nsample)+"x"
        +to_string(Nsimul)+"x"
        +to_string(dt)+"xSeed"
        +to_string(seed)+".data",raw_ascii);
  cout << "FileSaved"<< endl;
  return 0;
}
