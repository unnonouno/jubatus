// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011 Preferred Infrastructure and Nippon Telegraph and Telephone Corporation.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License version 2.1 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef JUBATUS_CORE_STORAGE_LOCAL_STORAGE_MIXTURE_HPP_
#define JUBATUS_CORE_STORAGE_LOCAL_STORAGE_MIXTURE_HPP_

#include <stdint.h>
#include <map>
#include <string>
#include <vector>
#include "jubatus/util/data/serialization.h"
#include "jubatus/util/data/serialization/unordered_map.h"
#include "jubatus/util/data/intern.h"
#include "local_storage.hpp"
#include "../common/version.hpp"

namespace jubatus {
namespace core {
namespace storage {

class local_storage_mixture : public storage_base {
 public:
  local_storage_mixture();
  ~local_storage_mixture();

  void get(const std::string& feature, feature_val1_t& ret) const;
  void get2(const std::string& feature, feature_val2_t& ret) const;
  void get3(const std::string& feature, feature_val3_t& ret) const;

  /// inner product
  void inp(const common::sfv_t& sfv, map_feature_val1_t& ret) const;

  void get_diff(diff_t& ret) const;
  bool set_average_and_clear_diff(const diff_t& average);

  void set(
      const std::string& feature,
      const std::string& klass,
      const val1_t& w);
  void set2(
      const std::string& feature,
      const std::string& klass,
      const val2_t& w);
  void set3(
      const std::string& feature,
      const std::string& klass,
      const val3_t& w);

  void get_status(std::map<std::string, std::string>& status) const;

  void update(
      const std::string& feature,
      const std::string& inc_class,
      const std::string& dec_class,
      const val1_t& v);

  void bulk_update(
      const common::sfv_t& sfv,
      float step_width,
      const std::string& inc_class,
      const std::string& dec_class);

  void register_label(const std::string& label);
  void clear();
  std::vector<std::string> get_labels() const;
  bool set_label(const std::string& label);

  void pack(msgpack::packer<msgpack::sbuffer>& packer) const;
  void unpack(msgpack::object o);

  version get_version() const {
    return model_version_;
  }

  std::string type() const;

  MSGPACK_DEFINE(tbl_, class2id_, tbl_diff_, model_version_);

 private:
  friend class jubatus::util::data::serialization::access;
  template <class Ar>
  void serialize(Ar& ar) {
    ar
      & JUBA_MEMBER(tbl_)
      & JUBA_MEMBER(class2id_)
      & JUBA_MEMBER(tbl_diff_)
      & JUBA_MEMBER(model_version_);
  }

  bool get_internal(const std::string& feature, id_feature_val3_t& ret) const;

  id_features3_t tbl_;
  common::key_manager class2id_;
  id_features3_t tbl_diff_;
  version model_version_;
};

}  // namespace storage
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_STORAGE_LOCAL_STORAGE_MIXTURE_HPP_
