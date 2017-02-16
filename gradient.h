#ifndef GRADIENT_H_
#define GRADIENT_H_

/*
 * A class which implements Stochastic Gradient Descent.
 * Features:
 * - Specify number of samples to train per time-step.
 * - Momentum strategy implemented allowing for faster convergence.
 */

 #include "network.h"
 #include <unordered_set>

class StochasticGradientDescent {
  NeuralNetwork * network;
  double trainingRate;
  int maxItterations;
  std::vector<boost::numeric::ublas::matrix<double> > velocity;
  double momentum;
  bool enableMomentum;
public:
  StochasticGradientDescent(NeuralNetwork * _net, const double rate = 0.01, const int _max = 20000, const double _momentum = 0.9, const bool _enableMomentum = true) {
    network = _net;
    trainingRate = rate;
    maxItterations = _max;
    momentum = _momentum;
    enableMomentum = _enableMomentum;
  }

  void train(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected, const double minCost, const int batchSize = 0);
};

#endif
