// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2012 Preferred Infrastructure and Nippon Telegraph and Telephone Corporation.
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

#include "regexp_splitter.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "exception.hpp"

namespace jubatus {
namespace core {
namespace fv_converter {

TEST(regexp_splitter, trivial) {
  regexp_splitter r("([0-9]+)", 1);

  std::vector<std::pair<size_t, size_t> > bs;
  r.split("aaa012bbb12ccc", bs);

  ASSERT_EQ(2u, bs.size());

  EXPECT_EQ(3u, bs[0].first);
  EXPECT_EQ(3u, bs[0].second);

  EXPECT_EQ(9u, bs[1].first);
  EXPECT_EQ(2u, bs[1].second);
}

TEST(regexp_splitter, end) {
  regexp_splitter r("/([^/]+)/", 1);
  std::vector<std::pair<size_t, size_t> > bs;
  r.split("/hoge/fuga/foo/hogee", bs);
  ASSERT_EQ(3u, bs.size());
  EXPECT_EQ(1u, bs[0].first);
  EXPECT_EQ(4u, bs[0].second);
  EXPECT_EQ(6u, bs[1].first);
  EXPECT_EQ(4u, bs[1].second);
  EXPECT_EQ(11u, bs[2].first);
  EXPECT_EQ(3u, bs[2].second);
}

TEST(regexp_splitter, match_empty) {
  regexp_splitter r("().", 1);
  std::vector<std::pair<size_t, size_t> > bs;
  r.split("test", bs);
}

TEST(regexp_splitter, error) {
  EXPECT_THROW(regexp_splitter("[", 0), converter_exception);
  EXPECT_THROW(regexp_splitter("(.+)", 2), converter_exception);
  EXPECT_THROW(regexp_splitter("(.+)", -1), converter_exception);
}

#if 0
TEST(regexp_splitter, create_error) {
  std::map<std::string, std::string> p;
  EXPECT_THROW(create(p), converter_exception);
  p["pattern"] = "(.+)";
  p["group"] = "a";
  EXPECT_THROW(create(p), converter_exception);
}
#endif

}  // namespace fv_converter
}  // namespace core
}  // namespace jubatus
