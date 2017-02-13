#ifndef NETWORK_H_
#define NETWORK_H_

#include "activation.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>
#include <chrono>
#include <deque>
#include <vector>

class NeuralNetwork {
private:
  int numberInput; // Number of input neurons
  int numberOutput; // Number of output neurons
  std::vector<boost::numeric::ublas::matrix <double> > weights;
  ActivationFunction * activation;

public:
  NeuralNetwork(std::vector<int> layerSize, int input, int output) {
    numberOutput = output;
    numberInput = input;
    activation = new SigmoidFunction();

    // Add input weights
    weights.push_back(boost::numeric::ublas::matrix<double>(layerSize[0], input + 1));

    // Add weights for hidden layers
    for (int i = 1; i < layerSize.size(); ++i) {
        weights.push_back(boost::numeric::ublas::matrix<double>(layerSize[i], layerSize[i - 1] + 1));
    }

    // Add output weights
    weights.push_back(boost::numeric::ublas::matrix<double>(output, layerSize[layerSize.size() - 1] + 1));
  }

  void initializeRandomWeights(double epsilon = 0.12);
  boost::numeric::ublas::vector<double> feedForwardVector(boost::numeric::ublas::vector<double> input);
  std::vector<boost::numeric::ublas::matrix<double> > backPropogateVector(boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> expected);
  boost::numeric::ublas::vector<double> addBiasUnit(boost::numeric::ublas::vector<double> input);

  ~NeuralNetwork() {
    delete activation;
  }

  std::vector<boost::numeric::ublas::matrix<double> > getWeights() {
    return weights;
  }

  void setWeights(std::vector<boost::numeric::ublas::matrix<double> > newWeights) {
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
