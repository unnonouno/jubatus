#pragma once

#include "classifier_base.hpp"

namespace jubatus {

class logistic_regression : public classifier_base {
 public:
  logistic_regression(storage::storage_base* storage);

  void train(const sfv_t& fv, const std::string& label);
  std::string name() const;

 private:
  void update(const sfv_t& fv, float w, const std::string& label);

  double learning_rate_;
};

}
