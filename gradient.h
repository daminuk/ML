#ifndef GRADIENT_H_
#define GRADIENT_H_

/*
 * A class which implements Stochastic Gradient Descent.
 */

 #include "network.h"
 #include <set>

class StochasticGradientDescent {
  NeuralNetwork * network;
  double trainingRate;
  int maxItterations;
public:
  StochasticGradientDescent(NeuralNetwork * _net, const double rate = 0.01, const int _max = 20000) {
    network = _net;
    trainingRate = rate;
    maxItterations = _max;
  }

  void train(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected, const double minCost, const int batchSize = 0);
};

#endif
