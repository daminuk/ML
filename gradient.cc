#include "gradient.h"

void StochasticGradientDescent::train(const std::vector<boost::numeric::ublas::vector<double>> input, const std::vector<boost::numeric::ublas::vector<double> > expected, double minCost) {
  /*
   * This function trains a network using SGD: we reduce the cost function until the
   * desired accuracy or the maximum number of itterations is reached.
   */

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::uniform_int_distribution<int> distribution(0, input.size() - 1);

  int select;
  int itt = 0;
  double J = network->cost(input, expected);

  while (J > minCost && itt < maxItterations) {
    select = distribution(generator);

    auto weights = network->getWeights();
    auto derivative = network->backPropogateVector(input[select], expected[select]);

    for (int i = 0; i < weights.size(); ++i) {
      weights[i] = weights[i] - trainingRate*derivative[i];
    }

    network->setWeights(weights);
    J = network->cost(input, expected);
    itt++;
  }
}
