#include "gradient.h"

void StochasticGradientDescent::train(boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> expected) {
  auto derivative = network->backPropogateVector(input, expected);
  auto weights = network->getWeights();

  // We now loop over the weights subtracting the derivative times the learning rate.
  for (int i = 0; i < weights.size(); ++i) {
    weights[i] = weights[i] - trainingRate*derivative[i];
  }

  network->setWeights(weights);
}
