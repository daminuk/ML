#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <cmath>

class ActivationFunction {
public:
  virtual double activation(const double input) = 0;
  virtual double gradient(const double input) = 0;

  virtual ~ActivationFunction() {};
};

class SigmoidFunction: public ActivationFunction {
public:
  double activation(const double input) {
    return (1.0)/(1.0 + exp(-input));
  }

  double gradient(const double input) {
    return activation(input)*(1.0-activation(input));
  }
};

class LinearFunction: public ActivationFunction {
public:
  double activation(const double input) {
    return input;
  }

  double gradient(const double input) {
    return 1.0;
  }
};
#endif
