#include "logistic_regression.hpp"

#include <cmath>

using namespace std;

namespace jubatus {

namespace {

double log_sum_exp(double x, double y) {
  if (x > y) {
    return log_sum_exp(y, x);
  } else {
    return y + log(1.0 + exp(x - y));
  }
}

double log_sum_exp(const classify_result& scores) {
  double sum = scores[0].score;
  for (size_t i = 1; i < scores.size(); ++i) {
    sum = log_sum_exp(scores[i].score, sum);
  }
  return sum;
}

}

logistic_regression::logistic_regression(storage::storage_base* storage)
    : classifier_base(storage),
      learning_rate_(0.1) {}

void logistic_regression::train(const sfv_t& fv, const std::string& label) {
  classify_result scores;
  classify_with_scores(fv, scores);
  if (scores.empty()) {
    update(fv, 1, label);
    return;
  }

  double sum = log_sum_exp(scores);
  bool correct_label_found = false;
  for (size_t i = 0; i < scores.size(); ++i) {
    double prob = exp(scores[i].score - sum);
    double w;
    if (scores[i].label == label) {
      correct_label_found = true;
      w = 1 - prob;
    } else
      w = -prob;
    update(fv, w, scores[i].label);
  }
  if (!correct_label_found) {
    update(fv, 1, label);
  }
}

void logistic_regression::update(const sfv_t& fv, float w, const std::string& label) {
  update_weight(fv, learning_rate_ * w, label, "");
}

std::string logistic_regression::name() const {
  return "logistic_regression";
}


}
