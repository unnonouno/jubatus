// This file is auto-generated from anomaly.idl(0.4.5-347-g86989a6) with jenerator version 0.5.2-19-gd01989b/user-error
// *** DO NOT EDIT ***

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <glog/logging.h>

#include "jubatus/core/common/exception.hpp"
#include "../../server/framework/aggregators.hpp"
#include "../../server/framework/proxy.hpp"
#include "anomaly_types.hpp"

namespace jubatus {

int run_proxy(int argc, char* argv[]) {
  try {
    jubatus::server::framework::proxy k(
        jubatus::server::framework::proxy_argv(argc, argv, "anomaly"));
    k.register_async_cht<2, bool>("clear_row",
        jubatus::util::lang::function<bool(bool, bool)>(
        &jubatus::server::framework::all_and));
    k.register_async_random<id_with_score, jubatus::core::fv_converter::datum>(
        "add");
    k.register_async_cht<2, float, jubatus::core::fv_converter::datum>("update",
        jubatus::util::lang::function<float(float, float)>(
        &jubatus::server::framework::pass<float>));
    k.register_async_cht<2, float, jubatus::core::fv_converter::datum>(
        "overwrite", jubatus::util::lang::function<float(float, float)>(
        &jubatus::server::framework::pass<float>));
    k.register_async_broadcast<bool>("clear",
        jubatus::util::lang::function<bool(bool, bool)>(
        &jubatus::server::framework::all_and));
    k.register_async_random<float, jubatus::core::fv_converter::datum>(
        "calc_score");
    k.register_async_broadcast<std::vector<std::string> >("get_all_rows",
        jubatus::util::lang::function<std::vector<std::string>(
        std::vector<std::string>, std::vector<std::string>)>(
        &jubatus::server::framework::concat<std::string>));
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
