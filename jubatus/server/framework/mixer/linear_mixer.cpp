// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011-2013 Preferred Infrastructure and Nippon Telegraph and Telephone Corporation.
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

#include "linear_mixer.hpp"

#include <map>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <msgpack.hpp>

#include "jubatus/util/concurrent/lock.h"
#include "jubatus/util/lang/bind.h"
#include "jubatus/util/lang/shared_ptr.h"
#include "jubatus/util/system/time_util.h"
#include "jubatus/util/system/syscall.h"
#include "jubatus/core/common/version.hpp"
#include "jubatus/core/common/exception.hpp"
#include "jubatus/core/framework/mixable.hpp"
#include "../../common/membership.hpp"
#include "../../common/mprpc/rpc_mclient.hpp"
#include "../../common/unique_lock.hpp"

using std::map;
using std::vector;
using std::string;
using std::stringstream;
using std::pair;
using std::make_pair;
using jubatus::core::common::byte_buffer;
using jubatus::core::storage::version;
using jubatus::util::concurrent::scoped_lock;
using jubatus::util::concurrent::scoped_rlock;
using jubatus::util::concurrent::scoped_wlock;
using jubatus::util::system::time::clock_time;
using jubatus::util::system::time::get_clock_time;

namespace jubatus {
namespace server {
namespace framework {
namespace mixer {
namespace {

class linear_communication_impl : public linear_communication {
 public:
  linear_communication_impl(
      const jubatus::util::lang::shared_ptr<common::lock_service>& zk,
      const string& type,
      const string& name,
      int timeout_sec,
      const pair<string, int>& my_id);

  size_t update_members();
  jubatus::util::lang::shared_ptr<common::try_lockable> create_lock();
  void get_diff(common::mprpc::rpc_result_object& a) const;
  void put_diff(
      const vector<byte_buffer>& a,
      common::mprpc::rpc_result_object& result) const;
  byte_buffer get_model();

  bool register_active_list() const {
    common::unique_lock lk(m_);
    register_active(*zk_.get(), type_, name_, my_id_.first, my_id_.second);
    return true;
  }

  bool unregister_active_list() const {
    common::unique_lock lk(m_);
    unregister_active(*zk_.get(), type_, name_, my_id_.first, my_id_.second);
    return true;
  }

 private:
  jubatus::util::lang::shared_ptr<server::common::lock_service> zk_;
  mutable jubatus::util::concurrent::mutex m_;
  const string type_;
  const string name_;
  const int timeout_sec_;
  const pair<string, int> my_id_;
  vector<pair<string, int> > servers_;
};

linear_communication_impl::linear_communication_impl(
    const jubatus::util::lang::shared_ptr<server::common::lock_service>& zk,
    const string& type,
    const string& name,
    int timeout_sec,
    const pair<string, int>& my_id)
    : zk_(zk),
      type_(type),
      name_(name),
      timeout_sec_(timeout_sec),
      my_id_(my_id) {
}

jubatus::util::lang::shared_ptr<common::try_lockable>
linear_communication_impl::create_lock() {
  string path;
  common::unique_lock lk(m_);
  common::build_actor_path(path, type_, name_);
  return jubatus::util::lang::shared_ptr<common::try_lockable>(
      new common::lock_service_mutex(*zk_, path + "/master_lock"));
}

size_t linear_communication_impl::update_members() {
  common::unique_lock lk(m_);
  common::get_all_nodes(*zk_, type_, name_, servers_);
#ifndef NDEBUG
  string members = "";
  for (size_t i = 0; i < servers_.size(); ++i) {
    members += "[" + servers_[i].first + ":"
        + jubatus::util::lang::lexical_cast<string>(servers_[i].second) + "] ";
  }
#endif
  return servers_.size();
}

byte_buffer linear_communication_impl::get_model() {
  update_members();
  for (;;) {
    common::unique_lock lk(m_);

    // use time as pseudo random number(it should enough)
    if (servers_.empty() || servers_.size() == 1) {
      return byte_buffer();
    }

    const jubatus::util::system::time::clock_time now(get_clock_time());
    const size_t target = now.usec % servers_.size();
    const string server_ip = servers_[target].first;
    const int server_port = servers_[target].second;
    if (server_ip == my_id_.first && server_port == my_id_.second) {
      // avoid get model from itself
      continue;
    }

    msgpack::rpc::client cli(server_ip, server_port);
    msgpack::rpc::future result(cli.call("get_model", 0));
    const byte_buffer got_model_data(result.get<byte_buffer>());
    LOG(INFO) << "got model(serialized data) " << got_model_data.size()
              << " from server[" << server_ip << ":" << server_port << "] ";
    return got_model_data;
  }
}

void linear_communication_impl::get_diff(
    common::mprpc::rpc_result_object& result) const {
  // TODO(beam2d): to be replaced to new client with socket connection pooling
  common::unique_lock lk(m_);
  common::mprpc::rpc_mclient client(servers_, timeout_sec_);

#ifndef NDEBUG
  for (size_t i = 0; i < servers_.size(); i++) {
    DLOG(INFO) << "get diff from " << servers_[i].first << ":"
               << servers_[i].second;
  }
#endif
  result = client.call("get_diff", 0);
}

void linear_communication_impl::put_diff(
    const vector<byte_buffer>& mixed,
    common::mprpc::rpc_result_object& result) const {
  common::unique_lock lk(m_);
  // TODO(beam2d): to be replaced to new client with socket connection pooling
  server::common::mprpc::rpc_mclient client(servers_, timeout_sec_);
#ifndef NDEBUG
  for (size_t i = 0; i < servers_.size(); i++) {
    DLOG(INFO) << "put diff to " << servers_[i].first << ":"
               << servers_[i].second;
  }
#endif
  lk.unlock();  // unlock for re-entrant lock aquisition over RPC
  result = client.call("put_diff", mixed);
}

string server_list(const vector<pair<string, uint16_t> >& servers) {
  stringstream out;
  for (size_t i = 0; i < servers.size(); ++i) {
    out << servers[i].first << ":" << servers[i].second;
    if (i + 1 < servers.size()) {
      out << ", ";
    }
  }
  return out.str();
}

string version_list(const std::vector<version>& versions)  {
  stringstream ss;
  ss << "[";
  for (size_t i = 0; i < versions.size(); ++i) {
    ss << versions[i];
    if (i < versions.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

// MessagePack-RPC server error (positive integer)
const unsigned int NO_METHOD_ERROR = 1;
const unsigned int ARGUMENT_ERROR = 2;

string create_error_string(const msgpack::object& error) {
  switch (error.type) {
    case msgpack::type::RAW:
      return error.as<string>();

    case msgpack::type::POSITIVE_INTEGER:
      switch (error.as<unsigned int>()) {
        case NO_METHOD_ERROR:
          return "no method error";
        case ARGUMENT_ERROR:
          return "argument error";
        default:
          {
            string msg = "unknown remote error (";
            msg += jubatus::util::lang::lexical_cast<string>(
                error.as<unsigned int>());
            msg += ")";
            return msg;
          }
      }

    case msgpack::type::NEGATIVE_INTEGER:
      // local errno(system error) carried as negative_integer
      {
        const int error_code = -error.as<int>();
        string msg("system error: ");
        msg += jubatus::util::system::syscall::get_error_msg(error_code);
        msg += " (" +
          jubatus::util::lang::lexical_cast<string>(error_code) + ")";
        return msg;
      }

    default:
      return "unknown error";
  }
}

}  // namespace

jubatus::util::lang::shared_ptr<linear_communication>
linear_communication::create(
    const jubatus::util::lang::shared_ptr<server::common::lock_service>& zk,
    const string& type,
    const string& name,
    int timeout_sec,
    const pair<string, int>& my_id) {
  return jubatus::util::lang::shared_ptr<linear_communication_impl>(
      new linear_communication_impl(zk, type, name, timeout_sec, my_id));
}

linear_mixer::linear_mixer(
    jubatus::util::lang::shared_ptr<linear_communication> communication,
    unsigned int count_threshold,
    unsigned int tick_threshold)
    : communication_(communication),
      count_threshold_(count_threshold),
      tick_threshold_(tick_threshold),
      counter_(0),
      ticktime_(get_clock_time()),
      is_running_(false),
      is_obsolete_(true),
      t_(jubatus::util::lang::bind(&linear_mixer::stabilizer_loop, this)) {
}

linear_mixer::~linear_mixer() {
  stop();
}

void linear_mixer::register_api(rpc_server_t& server) {
  server.add<vector<byte_buffer>(int)>(  // NOLINT
      "get_diff",
      jubatus::util::lang::bind(
          &linear_mixer::get_diff, this, jubatus::util::lang::_1));
  server.add<int(vector<byte_buffer>)>(
      "put_diff",
      jubatus::util::lang::bind(&linear_mixer::put_diff,
                                this,
                                jubatus::util::lang::_1));
  server.add<byte_buffer(int)>(  // NOLINT
      "get_model",
      jubatus::util::lang::bind(&linear_mixer::get_model,
                                this,
                                jubatus::util::lang::_1));
  server.add<bool(void)>(  // NOLINT
      "do_mix",
      jubatus::util::lang::bind(&linear_mixer::do_mix,
                                this));
}

void linear_mixer::set_mixable_holder(
    jubatus::util::lang::shared_ptr<core::framework::mixable_holder> m) {
  mixable_holder_ = m;
}

void linear_mixer::start() {
  scoped_lock lk(m_);
  if (!is_running_) {
    is_running_ = true;
    t_.start();
  }
}

void linear_mixer::stop() {
  common::unique_lock lk(m_);
  if (is_running_) {
    is_running_ = false;
    lk.unlock();
    t_.join();
  }
}

bool linear_mixer::do_mix() {
  {
    common::unique_lock lk(m_);
    counter_ = 0;
    ticktime_ = get_clock_time();
  }
  try {
    LOG(INFO) << "forced to mix by user RPC";
    jubatus::util::lang::shared_ptr<common::try_lockable> zklock =
        communication_->create_lock();
    while (true) {
      if (zklock->try_lock()) {
        mix();
        return true;
      }
    }
  } catch (const jubatus::core::common::exception::jubatus_exception& e) {
    LOG(ERROR) << e.diagnostic_information(true);
  } catch (const std::exception& e) {
    LOG(WARNING) << "exception in mix: " << e.what();
  } catch (...) {
    LOG(ERROR) << "unexpected error";
  }
  return false;
}

void linear_mixer::updated() {
  scoped_lock lk(m_);
  ++counter_;
  if (counter_ >= count_threshold_
      || get_clock_time() - ticktime_ > tick_threshold_) {
    c_.notify();  // TODO(beam2d): need sync here?
  }
}

void linear_mixer::get_status(server_base::status_t& status) const {
  scoped_lock lk(m_);
  status["linear_mixer.count"] =
      jubatus::util::lang::lexical_cast<string>(counter_);
  // since last mix
  status["linear_mixer.ticktime"] =
      jubatus::util::lang::lexical_cast<string>(ticktime_.sec);
}

void linear_mixer::stabilizer_loop() {
  while (true) {
    jubatus::util::lang::shared_ptr<common::try_lockable> zklock =
        communication_->create_lock();
    try {
      common::unique_lock lk(m_);
      if (!is_running_) {
        return;
      }

      c_.wait(m_, 0.5);

      if (!is_running_) {
        return;
      }
      const clock_time new_ticktime = get_clock_time();
      if (((0 < count_threshold_ && counter_ >= count_threshold_)
          || (0 < tick_threshold_
              && new_ticktime - ticktime_ > tick_threshold_))
          && (0 < counter_)) {
        if (zklock->try_lock()) {
          LOG(INFO) << "getting zk_lock ok, starting mix:";
          counter_ = 0;
          ticktime_ = new_ticktime;

          lk.unlock();
          mix();

          // print versions of mixables
          LOG(INFO) << ".... mix done. versions"
                    << version_list(mixable_holder_->get_versions());
        }
      }

      if (is_obsolete_) {
        if (zklock->try_lock()) {
          if (is_obsolete_) {
            LOG(INFO) << "start to get model from other server";
            lk.unlock();
            update_model();
            mix();
          }
        } else {
          LOG(INFO) << "failed to get zklock, waiting..";
        }
      }
    } catch (const jubatus::core::common::exception::jubatus_exception& e) {
      LOG(ERROR) << e.diagnostic_information(true);
    } catch (const std::exception& e) {
      LOG(WARNING) << "stabilizer exception: " << e.what();
    } catch (...) {
      LOG(ERROR) << "unexpected error";
    }
  }
}

void linear_mixer::mix() {
  // this method is thread safe
  using jubatus::util::system::time::clock_time;
  using jubatus::util::system::time::get_clock_time;

  const clock_time start = get_clock_time();
  size_t s = 0;

  const size_t servers_size = communication_->update_members();
  if (servers_size == 0) {
    LOG(WARNING) << "no other server.";
    communication_->register_active_list();
    return;
  } else {
    try {
      core::framework::mixable_holder::mixable_list mixables =
          mixable_holder_->get_mixables();

      vector<vector<byte_buffer> > diffs;
      {  // get_diff() -> diffs
        common::mprpc::rpc_result_object result;
        communication_->get_diff(result);

        // convert from rpc_result_object to vector<vector<bite_buffer> >
        typedef pair<string, uint16_t> server;
        vector<server> successes;
        for (size_t i = 0; i < result.response.size(); ++i) {
          if (result.response[i].has_error()) {
            const string error_text(
                create_error_string(result.response[i].error()));
            LOG(WARNING) << "get_diff failed at "
                         << result.error[i].host() << ":"
                         << result.error[i].port()
                         << " : " << error_text;
            continue;
          }
          diffs.push_back(result.response[i].as<vector<byte_buffer> >());
          successes.push_back(
              make_pair(result.error[i].host(), result.error[i].port()));
        }

        {  // success info message
          LOG(INFO) << "success to get_diff from ["
                    << server_list(successes) << "]";
        }
      }

      if (diffs.empty()) {
        throw JUBATUS_EXCEPTION(
            core::common::exception::runtime_error("no diff available"));
      }

      vector<byte_buffer> mixed = diffs.front();
      diffs.erase(diffs.begin());
      {  // diffs -(mix)-> mixed
        // it's doing foldr on diffs
        for (size_t i = 0; i < diffs.size(); ++i) {
          if (diffs[i].size() != mixed.size()) {
            throw JUBATUS_EXCEPTION(
                core::common::exception::runtime_error(
                    "got mixables length is invalid"));
          }
          for (size_t j = 0; j < diffs[i].size(); ++j) {
            mixables[j]->mix(diffs[i][j], mixed[j], mixed[j]);
          }
        }
      }

      {  // do put_diff
        common::mprpc::rpc_result_object result;
        communication_->put_diff(mixed, result);

        {  // log output
          for (size_t i = 0; i < mixed.size(); ++i) {
            s += mixed[i].size();
          }

          typedef pair<string, uint16_t> server;
          vector<server> successes;
          for (size_t i = 0; i < result.response.size(); ++i) {
            if (result.response[i].has_error()) {
              const string error_text(
                  create_error_string(result.response[i].error()));
              LOG(WARNING) << "put_diff failed at "
                           << result.error[i].host() << ":"
                           << result.error[i].port()
                           << " : " << error_text;
              continue;
            }
            successes.push_back(
              make_pair(result.error[i].host(), result.error[i].port()));
          }
          LOG(INFO) << "success to put_diff to ["
                    << server_list(successes) << "]";
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "mix failed :" << e.what();
      return;
    }
  }

  {
    const clock_time finish = get_clock_time();
    LOG(INFO) << "mixed with " << servers_size << " servers in "
              << static_cast<double>(finish - start) << " secs, " << s
              << " bytes (serialized data) has been put.";
  }
}


vector<byte_buffer> linear_mixer::get_diff(int a) {
  scoped_rlock lk_read(mixable_holder_->rw_mutex());
  scoped_lock lk(m_);

  core::framework::mixable_holder::mixable_list mixables =
      mixable_holder_->get_mixables();
  if (mixables.empty()) {
    throw JUBATUS_EXCEPTION(core::common::config_not_set());  // nothing to mix
  }

  vector<byte_buffer> o;
  for (size_t i = 0; i < mixables.size(); ++i) {
    o.push_back(mixables[i]->get_diff());
  }
  return o;
}

byte_buffer linear_mixer::get_model(int a) const {
  scoped_rlock lk_read(mixable_holder_->rw_mutex());
  msgpack::sbuffer packed;
  msgpack::packer<msgpack::sbuffer> pk(packed);
  mixable_holder_->pack(pk);

  LOG(INFO) << "sending learning-model. size = "
            << jubatus::util::lang::lexical_cast<string>(packed.size());

  return byte_buffer(packed.data(), packed.size());
}

void linear_mixer::update_model() {
  const byte_buffer model_serialized = communication_->get_model();

  if (model_serialized.size() == 0) {
    // it means "no other server"
    LOG(INFO) << "no other server available, I become active";
    is_obsolete_ = false;
    communication_->register_active_list();
    return;
  }
  msgpack::unpacked unpacked;
  msgpack::unpack(&unpacked, model_serialized.ptr(), model_serialized.size());
  {
    scoped_wlock lk_write(mixable_holder_->rw_mutex());
    mixable_holder_->unpack(unpacked.get());
  }
}

int linear_mixer::put_diff(
    const vector<byte_buffer>& unpacked) {
  scoped_wlock lk_write(mixable_holder_->rw_mutex());
  scoped_lock lk(m_);

  core::framework::mixable_holder::mixable_list mixables =
      mixable_holder_->get_mixables();
  if (unpacked.size() != mixables.size()) {
    // deserialization error
    return -1;
  }

  size_t total_size = 0;
  bool not_obsolete = true;
  for (size_t i = 0; i < mixables.size(); ++i) {
    // put_diff() returns true if put_diff succeeded
    not_obsolete = mixables[i]->put_diff(unpacked[i]) && not_obsolete;
    total_size += unpacked[i].size();
  }

  // print versions of mixables
  const string versions = version_list(mixable_holder_->get_versions());

  // if all put_diff returns true, this model is not obsolete
  if (not_obsolete) {
    if (is_obsolete_) {  // if it was obsolete, register as active
      LOG(INFO) << "put_diff with " << total_size << " bytes finished "
                << "I got latest model. So I become active. "
                << "versions " << versions;
      communication_->register_active_list();
    } else {
      LOG(INFO) << "put_diff with " << total_size << " bytes finished "
                << "my model is still up to date. "
                << "versions " << versions;
    }
  } else {
    if (!is_obsolete_) {  // it it was not obslete, delete from active list
      LOG(INFO) << "put_diff with " << total_size << " bytes finished "
                << "I'm obsolete. I become inactive. "
                << "versions " << versions;
      communication_->unregister_active_list();
    } else {
      LOG(INFO) << "put_diff with " << total_size << " bytes finished "
                << "my model is still obsolete. "
                << "versions " << versions;
    }
  }
  is_obsolete_ = !not_obsolete;

  counter_ = 0;
  ticktime_ = get_clock_time();
  return 0;
}

}  // namespace mixer
}  // namespace framework
}  // namespace server
}  // namespace jubatus
