
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the non-publicly visible part of the tuner. It contains the header file for
// the TunerImpl class, the implemenation in the Pimpl idiom. This class contains a vector of
// KernelInfo objects, holding the actual kernels and parameters. This class interfaces between
// them. This class is also responsible for the actual tuning and the collection and dissemination
// of the results.
//
// -------------------------------------------------------------------------------------------------
//
// Copyright 2014 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

#ifndef CLTUNE_TUNER_IMPL_H_
#define CLTUNE_TUNER_IMPL_H_

// Uses either the OpenCL or CUDA back-end (CLCudaAPI C++11 headers)
#if USE_OPENCL
  #include "internal/clpp11.h"
#else
  #include "internal/cupp11.h"
#endif

#include "internal/kernel_info.h"

#include <string> // std::string
#include <vector> // std::vector
#include <memory> // std::shared_ptr
#include <complex> // std::complex
#include <stdexcept> // std::runtime_error

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class TunerImpl {
 // Note that everything here is public because of the Pimpl-idiom
 public:

  // Parameters
  static constexpr auto kMaxL2Norm = 1e-4; // This is the threshold for 'correctness'

  // Messages printed to stdout (in colours)
  static const std::string kMessageFull;
  static const std::string kMessageHead;
  static const std::string kMessageRun;
  static const std::string kMessageInfo;
  static const std::string kMessageVerbose;
  static const std::string kMessageOK;
  static const std::string kMessageWarning;
  static const std::string kMessageFailure;
  static const std::string kMessageResult;
  static const std::string kMessageBest;

  // Helper structure to hold the results of a tuning run
  struct TunerResult {
    std::string kernel_name;
    float time;
    size_t threads;
    bool status;
    KernelInfo::Configuration configuration;
  };

  // Initialize either with platform 0 and device 0 or with a custom platform/device
  explicit TunerImpl();
  explicit TunerImpl(size_t platform_id, size_t device_id);
  ~TunerImpl();

  // Starts the tuning process. This function is called directly from the Tuner API.
  void Tune();

  // Compiles and runs a kernel and returns the elapsed time
  TunerResult RunKernel(const std::string &source, const KernelInfo &kernel,
                        const size_t configuration_id, const size_t num_configurations);

  // Wrapper for the above method, which can be called from public API.
  void RunSingleKernel(const size_t id,
                       const std::vector<std::pair<std::string, size_t>> &parameter_values);

  // Copies an output buffer
  template <typename T> KernelInfo::MemArgument CopyOutputBuffer(KernelInfo::MemArgument &argument);

  // Stores the output of the reference run into the host memory
  void StoreReferenceOutput();
  template <typename T> void DownloadReference(KernelInfo::MemArgument &device_buffer);

  // Downloads the output of a tuning run and compares it against the reference run
  bool VerifyOutput();
  template <typename T> bool DownloadAndCompare(KernelInfo::MemArgument &device_buffer, const size_t i);
  template <typename T> double AbsoluteDifference(const T reference, const T result);

  // Trains and uses a machine learning model based on the search space explored so far
  void ModelPrediction(const Model model_type, const float validation_fraction,
                       const size_t test_top_x_configurations);

  // Prints results of a particular kernel run
  void PrintResult(FILE* fp, const TunerResult &result, const std::string &message) const;

  // Loads a file from disk into a string
  std::string LoadFile(const std::string &filename);

  // Prints a header of a new section in the tuning process
  void PrintHeader(const std::string &header_name) const;

  // Specific implementations of the helper structure to get the memory-type based on a template
  // argument. Supports all enumerations of MemType.
  template <typename T> MemType GetType();

  // Accessors to device data-types
  const Device device() const { return device_; }
  const Context context() const { return context_; }
  Queue queue() const { return queue_; }

  // Device variables
  Platform platform_;
  Device device_;
  Context context_;
  Queue queue_;

  // Settings
  size_t num_runs_; // This is used for more-accurate execution time measurement
  bool has_reference_;
  bool suppress_output_;
  bool output_search_process_;
  std::string search_log_filename_;

  // The search method and its arguments
  SearchMethod search_method_;
  std::vector<double> search_args_;

  // Verification technique settings
  VerificationTechnique verification_technique_;
  double tolerance_treshold_;

  // Storage of kernels and output copy buffers
  std::vector<KernelInfo> kernels_;
  std::vector<KernelInfo::MemArgument> arguments_output_copy_; // these may be modified by the kernel

  // Storage for the reference kernel and output
  std::unique_ptr<KernelInfo> reference_kernel_;
  std::vector<void*> reference_outputs_;

  // List of tuning results
  std::vector<TunerResult> tuning_results_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_TUNER_IMPL_H_
#endif
