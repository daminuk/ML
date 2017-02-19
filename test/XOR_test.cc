#define BOOST_TEST_MODULE XOR Test
#include <boost/test/included/unit_test.hpp>

#include "../network.h"
#include "../gradient.h"
#include "../evolution.h"
#include <memory>

// Setup the test parameters
struct XORdata {
  std::vector<boost::numeric::ublas::vector<double> > input;
  std::vector<boost::numeric::ublas::vector<double> > expected;

  XORdata() {
    boost::numeric::ublas::vector<double> in1(2);
    in1[0] = 0.0;
    in1[1] = 0.0;
    input.push_back(in1);

    boost::numeric::ublas::vector<double> expv1(1);
    expv1[0] = 0.0;
    expected.push_back(expv1);

    boost::numeric::ublas::vector<double> in2(2);
    in2[0] = 0.0;
    in2[1] = 1.0;
    input.push_back(in2);

    boost::numeric::ublas::vector<double> expv2(1);
    expv2[0] = 1.0;
    expected.push_back(expv2);

    boost::numeric::ublas::vector<double> in3(2);
    in3[0] = 1.0;
    in3[1] = 0.0;
    input.push_back(in3);

    boost::numeric::ublas::vector<double> expv3(1);
    expv3[0] = 1.0;
    expected.push_back(expv3);

    boost::numeric::ublas::vector<double> in4(2);
    in4[0] = 1.0;
    in4[1] = 1.0;
    input.push_back(in4);

    boost::numeric::ublas::vector<double> expv4(1);
    expv4[0] = 0.0;
    expected.push_back(expv4);
  }
};

BOOST_AUTO_TEST_CASE(XOR_test_feed_forward)
{
  /*
  * We test if the feed forward implementation is working by using known
  * weights for an XOR network.
  */

  XORdata test;

  std::vector<int> size;
  size.push_back(2);
  NeuralNetwork network(size, 2, 1, new SigmoidFunction());
  network.initializeRandomWeights();

  auto weights = network.getWeights();

  // Weights which produce an XOR network with the sigmoid activation function
  weights[0](0,0) = -10.0;
  weights[0](0,1) = 20.0;
  weights[0](0,2) = 20.0;

  weights[0](1,0) = 30.0;
  weights[0](1,1) = -20.0;
  weights[0](1,2) = -20.0;

  weights[1](0,0) = -30.0;
  weights[1](0,1) = 20.0;
  weights[1](0,2) = 20.0;

  network.setWeights(weights);

  // We test the inputs against the known outputs
  for (int i = 0; i < test.input.size(); ++i) {
    BOOST_CHECK_SMALL(network.feedForwardVector(test.input[i])[0] - test.expected[i][0], 0.01);
  }
}

BOOST_AUTO_TEST_CASE(XOR_test_train_SGD)
{
  /*
  * We test the back propogation implementation by using stochastic gradient descent
  * to learn the weights for an XOR network.
  *
  * This test may fail if poor initial conditions are randomly selected.
  */

  XORdata test;

  std::vector<int> size;
  size.push_back(2);

  std::unique_ptr<NeuralNetwork> network(new NeuralNetwork(size, 2, 1, new SigmoidFunction()));
  network->initializeRandomWeights();

  StochasticGradientDescent SGD(network.get(), 0.1);
  // We train the network using the above pairs of inputs and expected values.
  std::cout << "Before training J=" << network->cost(test.input, test.expected) << std::endl;
  // We use SGD to train the weights.
  SGD.train(test.input, test.expected, 1e-3, 2);
  std::cout << "After training J=" << network->cost(test.input, test.expected) << std::endl;

  // We test the newly found weights
  for (int i = 0; i < test.input.size(); ++i) {
    std::cout << "Output: " << network->feedForwardVector(test.input[i])[0] << " Expected: " << test.expected[i][0] << std::endl;
    BOOST_CHECK_SMALL(network->feedForwardVector(test.input[i])[0] - test.expected[i][0], 0.01);
  }
}

BOOST_AUTO_TEST_CASE(XOR_test_train_evo)
{
  /*
  * We use the FEP implementation to learn weights for an XOR network.
  */

  XORdata test;

  std::vector<int> size;
  size.push_back(2);

  std::unique_ptr<NeuralNetwork> network(new NeuralNetwork(size, 2, 1, new SigmoidFunction()));
  network->initializeRandomWeights();

  EvolutionaryProgramming FEP(network.get(), -20.0, 20.0, 100);
  // We train the network using the above pairs of inputs and expected values.
  std::cout << "Before training J=" << network->cost(test.input, test.expected) << std::endl;
  std::cout << "Maximum number of fitness evaluations: " << 100000 << std::endl;

  // We use FEP to train the weights
  FEP.train(test.input, test.expected, 100000);
  std::cout << "After training J=" << network->cost(test.input, test.expected) << std::endl;

  // We test the newly found weights
  for (int i = 0; i < test.input.size(); ++i) {
    std::cout << "Output: " << network->feedForwardVector(test.input[i])[0] << " Expected: " << test.expected[i][0] << std::endl;
    BOOST_CHECK_SMALL(network->feedForwardVector(test.input[i])[0] - test.expected[i][0], 0.01);
  }
}
