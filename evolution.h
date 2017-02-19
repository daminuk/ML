#ifndef EVOLUTION_H_
#define EVOLUTION_H_

/*
 * An optimization class which implements Fast Evolutionary Programming for learning network weights.
 * Fast Evolutionary Programming is a global optimization technique which works well for multi-modal data.
 */

#include "network.h"
#include <unordered_set>

class EvolutionaryProgramming {
private:
  // Representation for each member of the population (contains the trained weights)
  struct Individual {
    std::vector<boost::numeric::ublas::matrix<double>> weights;
    std::vector<boost::numeric::ublas::matrix<double>> stepSize;
    double fitness;
    int wins;

    Individual():fitness(-1.0),
                 wins(0) {}
  };

  NeuralNetwork * network;
  double maxValue;
  double minValue;
  int populationSize;
  int fitnessEvaluations;
  std::vector<Individual> population;
  int dim;
  int opponentNumber;

  void generatePopulation();
  void spawnOffspring();
  double evaluateFitness(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected);
  void tournamentSelection();

public:
  EvolutionaryProgramming(NeuralNetwork * _net, double minVal, double maxVal, int popSize, int opNum = 10):
      network(_net), minValue(minVal), maxValue(maxVal), populationSize(popSize), opponentNumber(opNum),
      fitnessEvaluations(0), dim(0) {

    // Work out the dimensionality of the weights
    auto weights = network->getWeights();

    for (auto w : weights) {
      for_each(w.begin1(), w.end1(), [this] (double &val) {
        dim++;
      });
    }
  }

  void train(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected, int maxFitnessEval = 100000);
};

#endif
