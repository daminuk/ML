#include "gradient.h"

void StochasticGradientDescent::train(const std::vector<boost::numeric::ublas::vector<double>> input, const std::vector<boost::numeric::ublas::vector<double> > expected, const double minCost, const int batchSize) {
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

    auto weights = network->getWeights();

    // Add support for training a small subset of the data (minibatch)
    if (batchSize > 1) {

      // We keep a record of the elements which have already been learnt.
      std::set<int> learnt;

      // Return If the batchSize is larger than the dataset
      if (batchSize > input.size()) {
        return;
      }

      while (learnt.size() != batchSize) {

        // Find an element which we haven't previously trained
        select = distribution(generator);

        while (learnt.count(select)) {
          select = distribution(generator);
        }

        learnt.insert(select);
        auto derivative = network->backPropogateVector(input[select], expected[select]);

        for (int j = 0; j < weights.size(); ++j) {
          weights[j] = weights[j] - (trainingRate/batchSize)*derivative[j];
        }
      }

    } else {
      // We randomly select an element from the dataset for training.
      select = distribution(generator);

      auto weights = network->getWeights();
      auto derivative = network->backPropogateVector(input[select], expected[select]);

      for (int i = 0; i < weights.size(); ++i) {
        weights[i] = weights[i] - trainingRate*derivative[i];
      }

    }

    network->setWeights(weights);
    J = network->cost(input, expected);
    itt++;
  }
}
