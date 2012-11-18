// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011,2012 Preferred Infrastructure and Nippon Telegraph and Telephone Corporation.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

#pragma once

#include "classifier_base.hpp"

namespace jubatus {

class soft_confidence_weighted : public classifier_base {
 public:
  enum penalty_type {
    NONE, ONE, TWO
  };

  soft_confidence_weighted(storage::storage_base* storage,
                           penalty_type penalty = NONE);
  void train(const sfv_t& fv, const std::string& label);
  std::string name() const;

 private:
  double calc_alpha(double m, double v) const;
  void update(const sfv_t& fv, double alpha, double beta, const std::string& pos_label, const std::string& neg_label);

  penalty_type penalty_;
};

}
