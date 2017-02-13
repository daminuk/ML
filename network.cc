#include "network.h"

boost::numeric::ublas::vector<double> NeuralNetwork::feedForwardVector(boost::numeric::ublas::vector<double> input) {
  // If the input size does not match the networks return an empty vector
  if (input.size() != numberInput) {
    return boost::numeric::ublas::vector<double>();
  }

  boost::numeric::ublas::vector<double> current = input;

  for (auto w : weights) {
    boost::numeric::ublas::vector<double> tmp(w.size1());

    // We manually include the bias unit so to avoid resizing a vector.
    for (int i = 0; i < w.size1(); ++i) {
      // Add bias unit
      tmp[i] = 1.0 * w(i, 0);
      for (int j = 1; j < w.size2(); ++j) {
        tmp[i] += current[j-1] * w(i, j);
      }
    }

    current = tmp;

    // Apply activation function
    std::for_each(current.begin(), current.end(), [this] (double &val) {
      val = activation->activation(val);
    });
  }

  return current;
}

std::vector<boost::numeric::ublas::matrix<double> > NeuralNetwork::backPropogateVector(boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> expected) {
  // If the input size or output size does not match the networks return an empty vector
  if (input.size() != numberInput ||  expected.size() != numberOutput) {
    return std::vector<boost::numeric::ublas::matrix<double> >();
  }

  std::vector<boost::numeric::ublas::vector<double> > a;
  std::vector<boost::numeric::ublas::vector<double> > z;

  boost::numeric::ublas::vector<double> current = input;
  a.push_back(addBiasUnit(input));

  for (auto w : weights) {
    boost::numeric::ublas::vector<double> tmp(w.size1());

    // We manually include the bias unit so we don't need to resize a vector.
    for (int i = 0; i < w.size1(); ++i) {
      // Add bias unit
      tmp[i] = 1.0 * w(i, 0);
      for (int j = 1; j < w.size2(); ++j) {
        tmp[i] += current[j-1] * w(i, j);
      }
    }

    current = tmp;
    z.push_back(current);

    // Apply activation function
    std::for_each(current.begin(), current.end(), [this] (double &val) {
      val = activation->activation(val);
    });

    a.push_back(addBiasUnit(current));
  }

  std::deque<boost::numeric::ublas::vector<double> > delta;

  // Error from output layer and expected value
  delta.push_front(current - expected);

  // Propogate Error to other layers
  for (int k = weights.size() - 1; k > 0; k--) {

    boost::numeric::ublas::vector<double> currA = a[k];
    boost::numeric::ublas::vector<double> currZ = z[k-1];

    boost::numeric::ublas::vector<double> stepBack = boost::numeric::ublas::prod(boost::numeric::ublas::trans(weights[k]), delta.front());

    // Apply the gradient of the activation function
    std::for_each(currZ.begin(), currZ.end(), [this] (double &val) {
      val = activation->gradient(val);
    });

    // Resize tmp by removing the first element (bias unit)
    boost::numeric::ublas::vector<double> tmp2(stepBack.size()-1);

    for (int i = 1; i < stepBack.size(); ++i) {
      stepBack[i] *= currZ[i-1];
      tmp2[i-1] = stepBack[i];
    }

    // Add the caculated delta to the front
    delta.push_front(tmp2);
  }

  std::vector<boost::numeric::ublas::matrix<double> > Delta;
  // Finally we use the above deltas to calculate the weight derivates.
  for (int k=0; k < delta.size(); ++k) {
    Delta.push_back(boost::numeric::ublas::outer_prod(delta[k], a[k]));
  }

  return Delta;
}

boost::numeric::ublas::vector<double> NeuralNetwork::addBiasUnit(boost::numeric::ublas::vector<double> input) {
  // Add an element with value one to the start of a vector
  boost::numeric::ublas::vector<double> tmp(input.size() + 1);

  tmp[0] = 1.0;
  for (int j=0; j < input.size(); j++) {
      tmp[j+1] = input[j];
  }

  return tmp;
}

void NeuralNetwork::initializeRandomWeights(double epsilon) {
  // Initialize a normal distribution generator and
  // construct a trivial random generator engine from a time-based seed:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::normal_distribution<double> distribution(epsilon, 2.0 * epsilon);

  for (auto &w : weights) {
    for (int i = 0; i < w.size1(); ++i)
        for (int j = 0; j < w.size2(); ++j)
            w(i, j) = distribution(generator);
  }
}
