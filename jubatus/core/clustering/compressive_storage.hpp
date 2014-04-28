// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2013 Preferred Infrastructure and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_CLUSTERING_COMPRESSIVE_STORAGE_HPP_
#define JUBATUS_CORE_CLUSTERING_COMPRESSIVE_STORAGE_HPP_

#include <string>
#include <vector>
#include <msgpack.hpp>
#include "storage.hpp"
#include "compressor.hpp"

namespace jubatus {
namespace core {
namespace clustering {

class compressive_storage : public storage {
 public:
  compressive_storage(
      const std::string& name, const clustering_config& config);

  void add(const weighted_point& point);
  wplist get_mine() const;
  void set_compressor(
      jubatus::util::lang::shared_ptr<compressor::compressor> compressor);

  void pack(msgpack::packer<msgpack::sbuffer>& packer) const;
  void unpack(msgpack::object o);

  // hide storage::msgpack_pack and msgpack_unpack
  void msgpack_pack(msgpack::packer<msgpack::sbuffer>& packer) const {
    pack(packer);
  }
  void msgpack_unpack(msgpack::object o) {
    unpack(o);
  }

 private:
  struct coreset {
    wplist data;
    int carry_count;

    coreset() : data(), carry_count(0) {
    }

    MSGPACK_DEFINE(data, carry_count);
  };

  void carry_up(size_t r, wplist& carry);
  bool is_next_bucket_full(size_t bucket_number);
  bool reach_forgetting_threshold(size_t bucket_number);
  void forget_weight(wplist& points);
  void clear_mine();

  //std::vector<wplist> mine_;
  wplist lv0_;
  std::vector<coreset> coresets_;
  uint64_t status_;
  jubatus::util::lang::shared_ptr<compressor::compressor> compressor_;
};

}  // namespace clustering
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_CLUSTERING_COMPRESSIVE_STORAGE_HPP_
