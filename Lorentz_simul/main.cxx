#include <iostream>
#include "lorenz95.h"
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

int main()
{
  arma_rng::set_seed(0)
  string model = "L63";
  int Nsample = 100;
  int Nsimul = 50000;
  cube F = zeros(Nsimul, 3, Nsample);
  mat X = ones(3,Nsample)+randn(3,Nsample);
  for (int i=0; i<Nsimul; i=i+1){
    F.subcube(i, 0, 0, i , 2, Nsample-1)=X;
    M(X,1,model);
  }
  F.save("100x50000.data",raw_ascii);
  return 0;
}
