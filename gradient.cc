#include "gradient.h"

void StochasticGradientDescent::train(boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> expected) {
  std::vector<boost::numeric::ublas::matrix<double> > derivative = network->backPropogateVector(input, expected);
  std::vector<boost::numeric::ublas::matrix<double> > weights = network->getWeights();

  for (int i = 0; i < weights.size(); ++i) {
    weights[i] = weights[i] - trainingRate*derivative[i];
  }

  network->setWeights(weights);
}
