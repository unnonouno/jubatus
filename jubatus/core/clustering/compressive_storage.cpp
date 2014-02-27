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

#include "compressive_storage.hpp"

#include <string>
#include <vector>
#include "gmm_compressor.hpp"
#include "kmeans_compressor.hpp"

namespace jubatus {
namespace core {
namespace clustering {

compressive_storage::compressive_storage(
    const std::string& name,
    const clustering_config& config)
    : storage(name, config),
      status_(0) {
}

void compressive_storage::set_compressor(
    jubatus::util::lang::shared_ptr<compressor::compressor> compressor) {
  compressor_ = compressor;
}

void compressive_storage::add(const weighted_point& point) {
  wplist& c0 = lv0_;
  c0.push_back(point);
  if (c0.size() >= static_cast<size_t>(config_.bucket_size)) {
    wplist carry;
    compressor_->compress(
        c0, config_.bicriteria_base_size, config_.compressed_bucket_size, carry);
    c0.clear();
    status_ += 1;
    carry_up(0, carry);
    increment_revision();
  }
}

wplist compressive_storage::get_mine() const {
  wplist ret = lv0_;
  for (std::vector<coreset>::const_iterator it = coresets_.begin();
       it != coresets_.end(); ++it) {
    concat(it->data, ret);
  }
  return ret;
}

void compressive_storage::forget_weight(wplist& points) {
  double factor = std::exp(-config_.forgetting_factor);
  typedef wplist::iterator iter;
  for (iter it = points.begin(); it != points.end(); ++it) {
    it->weight *= factor;
  }
}

bool compressive_storage::reach_forgetting_threshold(size_t bucket_number) {
  double C = config_.forgetting_threshold;
  double lam = config_.forgetting_factor;
  if (std::exp(-lam * bucket_number) < C) {
    return true;
  }
  return false;
}

void compressive_storage::carry_up(size_t r, wplist& carry) {
  // insert `carry` to the r-th coreset.
  // if the r-th coreset is full, compress the set and the carry and insert it to the next.

  if (r >= coresets_.size()) {
    coresets_.push_back(coreset());
  }
  forget_weight(carry);
  coresets_[r].carry_count += 1;
  if (coresets_[r].carry_count < config_.bucket_length) {
    if (!reach_forgetting_threshold(r)) {
      concat(carry, coresets_[r].data);
    } else {
      coresets_[r].data.swap(carry);
    }
  } else {
    // As the r-th bucket is full, it needs to carry up.
    wplist current;
    coresets_[r].data.swap(current);
    concat(carry, current);
    coresets_[r].carry_count = 0;
    size_t dstsize = (r == 0) ? config_.compressed_bucket_size :
        2 * r * r * config_.compressed_bucket_size;
    wplist next_carry;
    compressor_->compress(current,
                          config_.bicriteria_base_size,
                          dstsize,
                          next_carry);
    carry_up(r + 1, next_carry);
  }
}

void compressive_storage::pack(
    msgpack::packer<msgpack::sbuffer>& packer) const {
  packer.pack_array(5);
  packer.pack(static_cast<const storage&>(*this));
  packer.pack(lv0_);
  packer.pack(coresets_);
  packer.pack(status_);
  packer.pack(*compressor_);
}

void compressive_storage::unpack(msgpack::object o) {
  std::vector<msgpack::object> mems;
  o.convert(&mems);
  if (mems.size() != 5) {
    throw msgpack::type_error();
  }
  mems[0].convert(static_cast<storage*>(this));
  mems[1].convert(&lv0_);
  mems[2].convert(&coresets_);
  mems[3].convert(&status_);
  mems[4].convert(compressor_.get());
}

void compressive_storage::clear_mine() {
  lv0_.clear();
  coresets_.clear();
  status_ = 0;
}

}  // namespace clustering
}  // namespace core
}  // namespace jubatus
