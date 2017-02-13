#ifndef GRADIENT_H_
#define GRADIENT_H_

/*
 * A class which carries out stochastic gradient descent.
 */

 #include "network.h"

class StochasticGradientDescent {
  NeuralNetwork * network;
  double trainingRate;
public:
  StochasticGradientDescent(NeuralNetwork * _net, double rate) {
    network = _net;
    trainingRate = rate;
  }

  void train(boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> expected);
};

#endif
