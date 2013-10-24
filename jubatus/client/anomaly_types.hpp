// This file is auto-generated from anomaly.idl
// *** DO NOT EDIT ***

#ifndef JUBATUS_CLIENT_ANOMALY_TYPES_HPP_
#define JUBATUS_CLIENT_ANOMALY_TYPES_HPP_

#include <stdint.h>

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <jubatus/client/common/datum.hpp>
#include <msgpack.hpp>

namespace jubatus {
namespace anomaly {

struct id_with_score {
 public:
  MSGPACK_DEFINE(id, score);
  std::string id;
  float score;
  id_with_score() {
  }
  id_with_score(const std::string& id, float score)
    : id(id), score(score) {
  }
};

}  // namespace anomaly
}  // namespace jubatus

#endif  // JUBATUS_CLIENT_ANOMALY_TYPES_HPP_
