#ifndef JUBATUS_FV_CONVERTER_TANH_FEATURE_HPP_
#define JUBATUS_FV_CONVERTER_TANH_FEATURE_HPP_

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "num_feature.hpp"

namespace jubatus {
namespace fv_converter {

class tanh_feature : public num_feature {
 public:
  tanh_feature(double center, double gradient)
      : center_(center),
        gradient_(gradient) {
  }

  void add_feature(const std::string& key, double value, sfv_t& ret_fv) const {
    double score = std::tanh((value - center_) / gradient_);
    ret_fv.push_back(make_pair(key, score));
  }

 private:
  const double center_;
  const double gradient_;
};

}  // fv_converter
}  // jubatus

#endif  // JUBATUS_FV_CONVERTER_TANH_FEATURE_HPP_
