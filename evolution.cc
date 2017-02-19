#include "evolution.h"

void EvolutionaryProgramming::generatePopulation() {
  /*
   * This function generates the initial population.
   */
   population.clear();

   auto weights = network->getWeights();

   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   std::default_random_engine generator (seed);
   std::uniform_real_distribution<double> distribution(minValue, maxValue);

   for (int i = 0; i < populationSize; ++i) {
     Individual tmp;
     tmp.weights = weights;
     tmp.stepSize = weights;

     // Randomly initialize weights
     for (auto &w : tmp.weights) {
       std::for_each(w.begin1(), w.end1(), [&] (double &val) {val = distribution(generator);});
     }

     // Set initial self-adaptive strategy parameter
     for (auto &step : tmp.stepSize) {
       std::for_each(step.begin1(), step.end1(), [] (double &val) {val = 3.0;});
     }

     population.push_back(tmp);
   }
}

void EvolutionaryProgramming::spawnOffspring() {
  /*
   * This function generates the offspring by random mutation (each member of
   * the population produces a single offspring).
   */

   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   std::default_random_engine generator (seed);
   std::normal_distribution<double> NormalDist(0.0, 1.0);
   std::cauchy_distribution<double> CauchyDist(0.0, 1.0);

   double stepRandom = NormalDist(generator);

   for (int i = 0; i < populationSize; ++i) {
     auto current = population[i];

     for (int weight = 0; weight < current.weights.size(); ++weight) {
       for (int x = 0; x < current.weights[weight].size1(); ++x) {
         for (int y = 0; y < current.weights[weight].size2(); ++y) {

           // Mutate the value of the matrix element using Cauchy random numbers
           double value = current.weights[weight](x,y) + current.stepSize[weight](x,y)*CauchyDist(generator);

           // Make sure the new value is within the desired bounds
           while (value < minValue || value > maxValue) {
             value = current.weights[weight](x,y) + current.stepSize[weight](x,y)*CauchyDist(generator);
           }

           // Assign mutated value
           current.weights[weight](x,y) = value;

           // Update the self-adaptive strategy parameter
           current.stepSize[weight](x,y) = current.stepSize[weight](x,y)*exp((1.0/sqrt(2.0*dim))*stepRandom + (1.0/sqrt(2.0*sqrt(dim)))*NormalDist(generator));
         }
       }
     }

     // Add the offspring to the population
     population.push_back(current);
   }
}

double EvolutionaryProgramming::evaluateFitness(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected) {
  /*
   * For all members of the population we calculate the fitness.
   */

   // We calculate the minimum fitness for this generation
   double minFit = -1;

   for (Individual &x: population) {
     // Set the network weights and use the cost function as the fitness
     network->setWeights(x.weights);
     x.fitness = network->cost(input, expected);

     if (minFit == -1 || x.fitness < minFit) {
       minFit = x.fitness;
     }

     fitnessEvaluations++;
   }

   return minFit;
}

void EvolutionaryProgramming::tournamentSelection() {
  /*
   * Carry out the tournament selection procedure: Individuals randomly compete with
   * one another and those with the largest number of wins make up the new population.
   */
   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   std::default_random_engine generator (seed);
   std::uniform_int_distribution<int> distribution(0, population.size() - 1);

   for (Individual &x: population) {

     x.wins = 0;

     std::unordered_set<int> seen;
     int select;

     // Make the current Individual compete with the desired number of opponents
     for (int i = 0; i < opponentNumber; ++i) {
       select = distribution(generator);

       while (seen.count(select)) {
         select = distribution(generator);
       }

       seen.insert(select);

       if (x.fitness <= population[select].fitness) {
         x.wins++;
       }
     }
   }

   // We sort the population by the number of wins (decreasing)
   std::sort(population.begin(), population.end(), [] (Individual i, Individual j) { return (i.wins > j.wins);});

   // Resize the population
   population.resize(populationSize);
}

void EvolutionaryProgramming::train(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected, int maxFitnessEval) {

  // Initialize population
  generatePopulation();

  int generation = 1;

  while (fitnessEvaluations < maxFitnessEval) {
    spawnOffspring();
    double fit = evaluateFitness(input, expected);
    tournamentSelection();
    generation++;
  }

  // We sort the final generation by fitness (increasing)
  std::sort(population.begin(), population.end(), [] (Individual i, Individual j) { return (i.fitness < j.fitness);});

  // Finally set the network weights to the best found
  network->setWeights(population[0].weights);
}
