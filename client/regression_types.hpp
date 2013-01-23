// This file is auto-generated from regression.idl
// *** DO NOT EDIT ***

#ifndef REGRESSION_TYPES_HPP_
#define REGRESSION_TYPES_HPP_

#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <stdint.h>
#include <msgpack.hpp>

namespace jubatus {

namespace regression {

struct datum {
 public:

  MSGPACK_DEFINE(string_values, num_values);

  std::vector<std::pair<std::string, std::string> > string_values;
  std::vector<std::pair<std::string, double> > num_values;
};

}  // namespace regression

}  // namespace jubatus

#endif // REGRESSION_TYPES_HPP_