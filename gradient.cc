#include "gradient.h"

void StochasticGradientDescent::train(const boost::numeric::ublas::vector<double> input, const boost::numeric::ublas::vector<double> expected) {
  /*
   * This function trains a network using SGD with the provided input/expected
   * value pair.
   */
   
  auto derivative = network->backPropogateVector(input, expected);
  auto weights = network->getWeights();

  // We now loop over the weights subtracting the derivative multiplied by the learning rate.
  for (int i = 0; i < weights.size(); ++i) {
    weights[i] = weights[i] - trainingRate*derivative[i];
  }

  network->setWeights(weights);
}
