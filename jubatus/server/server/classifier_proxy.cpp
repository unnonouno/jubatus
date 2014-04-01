// This file is auto-generated from classifier.idl(0.4.5-347-g86989a6) with jenerator version 0.5.2-19-gd01989b/user-error
// *** DO NOT EDIT ***

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <glog/logging.h>

#include "jubatus/core/common/exception.hpp"
#include "../../server/framework/aggregators.hpp"
#include "../../server/framework/proxy.hpp"
#include "classifier_types.hpp"

namespace jubatus {

int run_proxy(int argc, char* argv[]) {
  try {
    jubatus::server::framework::proxy k(
        jubatus::server::framework::proxy_argv(argc, argv, "classifier"));
    k.register_async_random<int32_t, std::vector<labeled_datum> >("train");
    k.register_async_random<std::vector<std::vector<estimate_result> >,
        std::vector<jubatus::core::fv_converter::datum> >("classify");
    k.register_async_random<std::vector<std::string> >("get_labels");
    k.register_async_random<bool, std::string>("set_label");
    k.register_async_broadcast<bool>("clear",
        jubatus::util::lang::function<bool(bool, bool)>(
        &jubatus::server::framework::all_and));
    return k.run();
  } catch (const jubatus::core::common::exception::jubatus_exception& e) {
    LOG(FATAL) << e.diagnostic_information(true);
    return -1;
  }
}

}  // namespace jubatus

int main(int argc, char* argv[]) {
  jubatus::run_proxy(argc, argv);
}
