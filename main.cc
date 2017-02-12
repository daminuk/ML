#include "network.h"

int main() {

  std::vector<int> size;
  size.push_back(10);
  size.push_back(5);
  size.push_back(10);

  NeuralNetwork network(size, 5, 5);
  network.initializeRandomWeights();

  boost::numeric::ublas::vector<double> tmp(5);

  for (int i=0; i < tmp.size(); ++i) {
    tmp[i] = i;
  }

  std::cout << "Output:- " <<  network.feedForwardVector(tmp) << std::endl;
  network.backPropogateVector(tmp, tmp);

};
