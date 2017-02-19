#ifndef NETWORK_H_
#define NETWORK_H_

#include "activation.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>
#include <chrono>
#include <deque>
#include <vector>
#include <memory>

class NeuralNetwork {
private:
  int numberInput; // Number of input neurons
  int numberOutput; // Number of output neurons
  std::vector<boost::numeric::ublas::matrix <double> > weights;
  std::unique_ptr<ActivationFunction> activation;

public:
  NeuralNetwork(const std::vector<int> layerSize, const int input, const int output, ActivationFunction* active):
    numberInput(input), numberOutput(output), activation(std::unique_ptr<ActivationFunction>(active)) {
    // Add input weights
    weights.push_back(boost::numeric::ublas::matrix<double>(layerSize[0], input + 1));

    // Add weights for hidden layers
    for (int i = 1; i < layerSize.size(); ++i) {
        weights.push_back(boost::numeric::ublas::matrix<double>(layerSize[i], layerSize[i - 1] + 1));
    }

    // Add output weights
    weights.push_back(boost::numeric::ublas::matrix<double>(output, layerSize[layerSize.size() - 1] + 1));
  }

  boost::numeric::ublas::vector<double> feedForwardVector(const boost::numeric::ublas::vector<double> input);
  std::vector<boost::numeric::ublas::matrix<double> > backPropogateVector(const boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> expected);
  boost::numeric::ublas::vector<double> addBiasUnit(const boost::numeric::ublas::vector<double> input);

  void initializeRandomWeights(const double epsilon = 0.12);
  double cost(const std::vector<boost::numeric::ublas::vector<double> > input, const std::vector<boost::numeric::ublas::vector<double> > expected);

  std::vector<boost::numeric::ublas::matrix<double> > getWeights() {
    return weights;
  }

  void setWeights(const std::vector<boost::numeric::ublas::matrix<double> > newWeights) {
    weights = newWeights;
  }

  int getInputSize() {
    return numberInput;
  }

  int getOutputSize() {
    return numberOutput;
  }


};

#endif
