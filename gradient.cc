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

    // If we are using the momentum strategy construct a velocity vector with
    // the same dimensions as the weights
    if (velocity.size() > 0 && enableMomentum) {
      for (int k = 0; k < velocity.size(); ++k) {
        velocity[k] = momentum*velocity[k];
      }
    } else if (enableMomentum) {
      for (int i = 0; i < weights.size(); ++i) {
        velocity.push_back(boost::numeric::ublas::matrix<double>(weights[i].size1(), weights[i].size2(), 0.0));
      }
    }

    // Add support for training a small subset of the data (minibatch)
    if (batchSize > 1) {

      // We keep a record of the elements which have already been learnt.
      std::unordered_set<int> seen;

      // Return If the batchSize is larger than the dataset
      if (batchSize > input.size()) {
        return;
      }

      while (seen.size() != batchSize) {

        // Find an element which we haven't previously trained
        select = distribution(generator);

        while (seen.count(select)) {
          select = distribution(generator);
        }

        seen.insert(select);
        auto derivative = network->backPropogateVector(input[select], expected[select]);

        if (enableMomentum) {
          for (int j = 0; j < weights.size(); ++j) {
            velocity[j] += (trainingRate/batchSize)*derivative[j];
          }
        } else {
          for (int j = 0; j < weights.size(); ++j) {
            weights[j] = weights[j] - (trainingRate/batchSize)*derivative[j];
          }
        }
      }

    } else {
      // We randomly select an element from the dataset for training.
      select = distribution(generator);

      auto weights = network->getWeights();
      auto derivative = network->backPropogateVector(input[select], expected[select]);

      if (enableMomentum) {
        for (int j = 0; j < weights.size(); ++j) {
          velocity[j] += (trainingRate/batchSize)*derivative[j];
        }
      } else {
        for (int i = 0; i < weights.size(); ++i) {
          weights[i] = weights[i] - trainingRate*derivative[i];
        }
      }

    }

    if (enableMomentum) {
      for (int i = 0; i < weights.size(); ++i) {
        weights[i] = weights[i] - velocity[i];
      }
      network->setWeights(weights);
    } else {
      network->setWeights(weights);
    }
    J = network->cost(input, expected);
    itt++;
  }
}
