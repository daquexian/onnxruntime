// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/capture.h"
#include "core/common/logging/logging.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace logging {

void Capture::CapturePrintf(msvc_printf_check const char* format, ...) {
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  va_list arglist;
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  va_start(arglist, format);
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;

  ProcessPrintf(format, arglist);
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;

  va_end(arglist);
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
}

// from https://github.com/KjellKod/g3log/blob/master/src/logcapture.cpp LogCapture::capturef
// License: https://github.com/KjellKod/g3log/blob/master/LICENSE
// Modifications Copyright (c) Microsoft.
void Capture::ProcessPrintf(msvc_printf_check const char* format, va_list args) {
  static constexpr auto kTruncatedWarningText = "[...truncated...]";
  static const int kMaxMessageSize = 2048;
  char message_buffer[kMaxMessageSize];
  const auto message = gsl::make_span(message_buffer);

  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  bool error = false;
  bool truncated = false;

  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__GNUC__))
  errno = 0;
  const int nbrcharacters = vsnprintf_s(message.data(), message.size(), _TRUNCATE, format, args);
  if (nbrcharacters < 0) {
    error = errno != 0;
    truncated = !error;
  }
#else
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  const int nbrcharacters = vsnprintf(message.data(), message.size(), format, args);
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  error = nbrcharacters < 0;
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  truncated = (nbrcharacters >= 0 && static_cast<gsl::index>(nbrcharacters) > message.size());
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
#endif

  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  if (error) {
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
    stream_ << "\n\tERROR LOG MSG NOTIFICATION: Failure to successfully parse the message";
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
    stream_ << '"' << format << '"' << std::endl;
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  } else if (truncated) {
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
    stream_ << message.data() << kTruncatedWarningText;
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  } else {
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
    stream_ << message.data();
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  }
}

Capture::~Capture() {
  if (logger_ != nullptr) {
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
    logger_->Log(*this);
  std::cout << __FILE__ << " " <<  __LINE__ << std::endl;
  }
}
}  // namespace logging
}  // namespace onnxruntime
