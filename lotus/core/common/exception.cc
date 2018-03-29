#include <functional>
#include "core/common/exceptions.h"

namespace Lotus {

static std::function<string(void)> FetchStackTrace = []() { return ""; };

EnforceNotMet::EnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const string& msg,
    const void* caller)
    : msg_stack_{MakeString(
          "[enforce fail at ",
          StripBasename(std::string(file)),
          ":",
          line,
          "] ",
          condition,
          ". ",
          msg,
          " ")},
      stack_trace_(FetchStackTrace()) {
  caller_ = caller;
  full_msg_ = this->Msg();
}

void EnforceNotMet::AppendMessage(const string& msg) {
  msg_stack_.push_back(msg);
  full_msg_ = this->Msg();
}

string EnforceNotMet::Msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), string("")) +
         stack_trace_;
}

const char* EnforceNotMet::what() const noexcept {
  return full_msg_.c_str();
}

const void* EnforceNotMet::Caller() const noexcept {
  return caller_;
}
}  // namespace Lotus
