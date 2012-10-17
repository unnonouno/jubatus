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

#include "soft_confidence_weighted.hpp"

#include <algorithm>
#include <cmath>
#include "classifier_util.hpp"

using namespace std;

namespace jubatus{

soft_confidence_weighted::soft_confidence_weighted(storage::storage_base* storage,
                                                   penalty_type penalty)
    : classifier_base(storage),
      penalty_(penalty)
{
  classifier_base::use_covars_ = true;
}

namespace {

double calc_alpha_0(double phi, double m, double v) {
  double pp = phi * phi;
  double psi = 1 + pp / 2;
  double zeta = 1 + pp;
  double w = (m * pp) / 2;
  double a = (-m * psi + sqrt(w * w + v * pp * zeta)) / (v * zeta);
  return max(0.0, a);
}

double calc_alpha_1(double C, double phi, double m, double v) {
  return min(C, calc_alpha_0(phi, m, v));
}

double calc_alpha_2(double C, double phi, double m, double v) {
  double pmv = phi * m * v;
  double n = v + 0.5 / C;
  double vpp = v * phi * phi;
  double gamma = phi * sqrt(pmv * pmv + 4 * n * v * (n + vpp));
  double a = (-m * (2 * n + vpp) + gamma) / (2 * n * (n + vpp));
  return max(0.0, a);
}

double calc_beta(double alpha, double v, double phi) {
  double w = alpha * v * phi;
  double u = fabs(-w + sqrt(w * w + 4 * v)) / 2;
  return (alpha * phi) / (u + w);
}

}

void soft_confidence_weighted::train(const sfv_t& sfv, const string& label) {
  string incorrect_label;
  float variance = 0.f;
  float margin = -calc_margin_and_variance(sfv, label, incorrect_label, variance);
  double alpha = calc_alpha(margin, variance);

  if (alpha <= 0.f) {
    return;
  }
  double beta = calc_beta(alpha, variance, C_);
  update(sfv, alpha, beta, label, incorrect_label);
}

double soft_confidence_weighted::calc_alpha(double m, double v) const {
  switch (penalty_) {
    case NONE:
      return calc_alpha_0(C_, m, v);
    case ONE:
      return calc_alpha_1(1.0, C_, m, v);
    case TWO:
      return calc_alpha_2(1.0, C_, m, v);
  }
}

void soft_confidence_weighted::update(const sfv_t& sfv, double alpha, double beta, const string& pos_label, const string& neg_label) {
  for (sfv_t::const_iterator it = sfv.begin(); it != sfv.end(); ++it) {
    const string& feature = it->first;
    float val = it->second;
    storage::feature_val2_t val2;
    storage_->get2(feature, val2);
    
    storage::val2_t pos_val(0.f, 1.f);
    storage::val2_t neg_val(0.f, 1.f);
    ClassifierUtil::get_two(val2, pos_label, neg_label, pos_val, neg_val);

    {    
      double pos = pos_val.v2 * val;
      float w = pos_val.v1 + alpha * pos;
      float var = pos_val.v2 - beta * pos * pos;
      storage_->set2(feature, pos_label, storage::val2_t(w, var));
    }
    if (neg_label != "") {
      double neg = neg_val.v2 * val;
      float w = neg_val.v1 - alpha * neg;
      float var = neg_val.v2 - beta * neg * neg;
      storage_->set2(feature, neg_label, storage::val2_t(w, var));
    }
  }
}

string soft_confidence_weighted::name() const{
  return string("soft_confidence_weighted");
}


}
