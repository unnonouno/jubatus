// This file is auto-generated from classifier.idl(0.4.5-347-g86989a6) with jenerator version 0.5.2-19-gd01989b/user-error
// *** DO NOT EDIT ***

#ifndef JUBATUS_CLIENT_CLASSIFIER_CLIENT_HPP_
#define JUBATUS_CLIENT_CLASSIFIER_CLIENT_HPP_

#include <map>
#include <string>
#include <vector>
#include <jubatus/client/common/client.hpp>
#include <jubatus/client/common/datum.hpp>
#include "classifier_types.hpp"

namespace jubatus {
namespace classifier {
namespace client {

class classifier : public jubatus::client::common::client {
 public:
  classifier(const std::string& host, uint64_t port, const std::string& name,
      unsigned int timeout_sec)
      : client(host, port, name, timeout_sec) {
  }

  int32_t train(const std::vector<labeled_datum>& data) {
    msgpack::rpc::future f = c_.call("train", name_, data);
    return f.get<int32_t>();
  }

  std::vector<std::vector<estimate_result> > classify(
      const std::vector<jubatus::client::common::datum>& data) {
    msgpack::rpc::future f = c_.call("classify", name_, data);
    return f.get<std::vector<std::vector<estimate_result> > >();
  }

  std::vector<std::string> get_labels() {
    msgpack::rpc::future f = c_.call("get_labels", name_);
    return f.get<std::vector<std::string> >();
  }

  bool set_label(const std::string& new_label) {
    msgpack::rpc::future f = c_.call("set_label", name_, new_label);
    return f.get<bool>();
  }

  bool clear() {
    msgpack::rpc::future f = c_.call("clear", name_);
    return f.get<bool>();
  }
};

}  // namespace client
}  // namespace classifier
}  // namespace jubatus

#endif  // JUBATUS_CLIENT_CLASSIFIER_CLIENT_HPP_
