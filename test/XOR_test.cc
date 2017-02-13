#define BOOST_TEST_MODULE XOR Test
#include <boost/test/included/unit_test.hpp>

#include "../network.h"
#include "../gradient.h"
#include <memory>

BOOST_AUTO_TEST_CASE(XOR_test_feed_forward)
{
  /*
   * We test if the feed forward implementation is working by using known
   * weights for an XOR network.
   */
  boost::numeric::ublas::vector<double> in1(2);
  in1[0] = 0.0;
  in1[1] = 0.0;

  boost::numeric::ublas::vector<double> expv1(1);
  expv1[0] = 0.0;

  boost::numeric::ublas::vector<double> in2(2);
  in2[0] = 0.0;
  in2[1] = 1.0;

  boost::numeric::ublas::vector<double> expv2(1);
  expv2[0] = 1.0;

  boost::numeric::ublas::vector<double> in3(2);
  in3[0] = 1.0;
  in3[1] = 0.0;

  boost::numeric::ublas::vector<double> expv3(1);
  expv3[0] = 1.0;

  boost::numeric::ublas::vector<double> in4(2);
  in4[0] = 1.0;
  in4[1] = 1.0;

  boost::numeric::ublas::vector<double> expv4(1);
  expv4[0] = 0.0;

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
  BOOST_CHECK_SMALL(network.feedForwardVector(in1)[0] - expv1[0], 0.01);
  BOOST_CHECK_SMALL(network.feedForwardVector(in2)[0] - expv2[0], 0.01);
  BOOST_CHECK_SMALL(network.feedForwardVector(in3)[0] - expv3[0], 0.01);
  BOOST_CHECK_SMALL(network.feedForwardVector(in4)[0] - expv4[0], 0.01);
}


BOOST_AUTO_TEST_CASE(XOR_test_train)
{
  /*
   * We test the back propogation implementation by using stochastic gradient descent
   * to learn the weights for an XOR network.
   *
   * This test may fail if poor initial conditions are randomly selected.
   */

  boost::numeric::ublas::vector<double> in1(2);
  in1[0] = 0.0;
  in1[1] = 0.0;

  boost::numeric::ublas::vector<double> expv1(1);
  expv1[0] = 0.0;

  boost::numeric::ublas::vector<double> in2(2);
  in2[0] = 0.0;
  in2[1] = 1.0;

  boost::numeric::ublas::vector<double> expv2(1);
  expv2[0] = 1.0;

  boost::numeric::ublas::vector<double> in3(2);
  in3[0] = 1.0;
  in3[1] = 0.0;

  boost::numeric::ublas::vector<double> expv3(1);
  expv3[0] = 1.0;

  boost::numeric::ublas::vector<double> in4(2);
  in4[0] = 1.0;
  in4[1] = 1.0;

  boost::numeric::ublas::vector<double> expv4(1);
  expv4[0] = 0.0;

  std::vector<int> size;
  size.push_back(2);
  std::unique_ptr<NeuralNetwork> network(new NeuralNetwork(size, 2, 1, new SigmoidFunction()));
  network->initializeRandomWeights();

  StochasticGradientDescent SGD(network.get(), 0.1);

  // We train the network using the above pairs of inputs and expected values.
  // Note: A criteria for halting the optimization is still needed.
  for (int i=0; i < 20000; ++i) {
    if (i % 100 == 0) {
    	std::cout << "J=" << network->costVector(in1, expv1) << std::endl;
    }
 
    SGD.train(in1, expv1);
    SGD.train(in2, expv2);
    SGD.train(in3, expv3);
    SGD.train(in4, expv4);
  }

  std::cout << "Output: " << network->feedForwardVector(in1)[0] << " Expected: " << expv1[0] << std::endl;
  std::cout << "Output: " << network->feedForwardVector(in2)[0] << " Expected: " << expv2[0] << std::endl;
  std::cout << "Output: " << network->feedForwardVector(in3)[0] << " Expected: " << expv3[0] << std::endl;
  std::cout << "Output: " << network->feedForwardVector(in4)[0] << " Expected: " << expv4[0] << std::endl;

  // We test the ouput of the newly trained network against known values.
  BOOST_CHECK_SMALL(network->feedForwardVector(in1)[0] - expv1[0], 0.1);
  BOOST_CHECK_SMALL(network->feedForwardVector(in2)[0] - expv2[0], 0.1);
  BOOST_CHECK_SMALL(network->feedForwardVector(in3)[0] - expv3[0], 0.1);
  BOOST_CHECK_SMALL(network->feedForwardVector(in4)[0] - expv4[0], 0.1);
}
