
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the TunerImpl class (see the header for information about the class).
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

// The corresponding header file
#include "internal/tuner_impl.h"

// The search strategies
#include "internal/searchers/full_search.h"
#include "internal/searchers/random_search.h"
#include "internal/searchers/annealing.h"
#include "internal/searchers/pso.h"

// The machine learning models
#include "internal/ml_models/linear_regression.h"
#include "internal/ml_models/neural_network.h"

#include <fstream> // std::ifstream, std::stringstream
#include <iostream> // FILE
#include <limits> // std::numeric_limits
#include <algorithm> // std::min
#include <memory> // std::unique_ptr
#include <tuple> // std::tuple

namespace cltune {
// =================================================================================================

// Messages printed to stdout (in colours for Unix)
#ifdef _WIN32
  const std::string TunerImpl::kMessageFull = "[==========]";
  const std::string TunerImpl::kMessageHead = "[----------]";
  const std::string TunerImpl::kMessageRun = "[ RUN      ]";
  const std::string TunerImpl::kMessageInfo = "[   INFO   ]";
  const std::string TunerImpl::kMessageVerbose = "[ VERBOSE  ]";
  const std::string TunerImpl::kMessageOK = "[       OK ]";
  const std::string TunerImpl::kMessageWarning = "[  WARNING ]";
  const std::string TunerImpl::kMessageFailure = "[   FAILED ]";
  const std::string TunerImpl::kMessageResult = "[ RESULT   ]";
  const std::string TunerImpl::kMessageBest = "[     BEST ]";
#else
  const std::string TunerImpl::kMessageFull = "\x1b[32m[==========]\x1b[0m";
  const std::string TunerImpl::kMessageHead = "\x1b[32m[----------]\x1b[0m";
  const std::string TunerImpl::kMessageRun = "\x1b[32m[ RUN      ]\x1b[0m";
  const std::string TunerImpl::kMessageInfo = "\x1b[32m[   INFO   ]\x1b[0m";
  const std::string TunerImpl::kMessageVerbose = "\x1b[39m[ VERBOSE  ]\x1b[0m";
  const std::string TunerImpl::kMessageOK = "\x1b[32m[       OK ]\x1b[0m";
  const std::string TunerImpl::kMessageWarning = "\x1b[33m[  WARNING ]\x1b[0m";
  const std::string TunerImpl::kMessageFailure = "\x1b[31m[   FAILED ]\x1b[0m";
  const std::string TunerImpl::kMessageResult = "\x1b[32m[ RESULT   ]\x1b[0m";
  const std::string TunerImpl::kMessageBest = "\x1b[35m[     BEST ]\x1b[0m";
#endif
  
// =================================================================================================

// Initializes the platform and device to the default 0
TunerImpl::TunerImpl():
    platform_(Platform(size_t{0})),
    device_(Device(platform_, size_t{0})),
    context_(Context(device_)),
    queue_(Queue(context_, device_)),
    num_runs_(size_t{1}),
    has_reference_(false),
    verification_method_(VerificationMethod::AbsoluteDifference),
    tolerance_treshold_(kMaxL2Norm),
    suppress_output_(false),
    output_search_process_(false),
    search_log_filename_(std::string{}) {
  if (!suppress_output_) {
    fprintf(stdout, "\n%s Initializing on platform 0 device 0\n", kMessageFull.c_str());
    auto opencl_version = device_.Version();
    auto device_name = device_.Name();
    fprintf(stdout, "%s Device name: '%s' (%s)\n", kMessageFull.c_str(),
            device_name.c_str(), opencl_version.c_str());
  }
}

// Initializes with a custom platform and device
TunerImpl::TunerImpl(size_t platform_id, size_t device_id):
    platform_(Platform(platform_id)),
    device_(Device(platform_, device_id)),
    context_(Context(device_)),
    queue_(Queue(context_, device_)),
    num_runs_(size_t{1}),
    has_reference_(false),
    verification_method_(VerificationMethod::AbsoluteDifference),
    tolerance_treshold_(kMaxL2Norm),
    suppress_output_(false),
    output_search_process_(false),
    search_log_filename_(std::string{}) {
  if (!suppress_output_) {
    fprintf(stdout, "\n%s Initializing on platform %zu device %zu\n",
            kMessageFull.c_str(), platform_id, device_id);
    auto opencl_version = device_.Version();
    auto device_name = device_.Name();
    fprintf(stdout, "%s Device name: '%s' (%s)\n", kMessageFull.c_str(),
            device_name.c_str(), opencl_version.c_str());
  }
}

// End of the tuner
TunerImpl::~TunerImpl() {
  for (auto &reference_output: reference_outputs_) {
    delete[] static_cast<int*>(reference_output);
  }

  // Frees the device buffers
  auto free_buffers = [](KernelInfo::MemArgument &mem_info) {
    #ifdef USE_OPENCL
      CheckError(clReleaseMemObject(mem_info.buffer));
    #else
      CheckError(cuMemFree(mem_info.buffer));
    #endif
  };
  for (auto &mem_argument: arguments_output_copy_) { free_buffers(mem_argument); }

  if (!suppress_output_) {
    fprintf(stdout, "\n%s End of the tuning process\n\n", kMessageFull.c_str());
  }
}

// =================================================================================================

// Wrapper for RunKernel() method, which can be called from public API.
PublicTunerResult TunerImpl::RunSingleKernel(const size_t id, const ParameterRange &parameter_values) {
  KernelInfo& kernel = kernels_.at(id);
  KernelInfo::Configuration configuration;

  PrintHeader("Running kernel " + kernel.name());

  if (parameter_values.size() > 0) {
    for (auto &parameter : parameter_values) {
      KernelInfo::Setting setting;
      setting.name = parameter.first;
      setting.value = parameter.second;
      configuration.push_back(setting);
    }

    // Adds the parameters to the source-code string as defines
    std::string source = GetConfiguredKernelSource(id, configuration);

    // Updates the local range with the parameter values
    kernel.ComputeRanges(configuration);

    // Updates number of kernel iterations based on parameter value
    kernel.SetNumCurrentIterations(configuration);
  }

  // Compiles and runs the kernel
  auto tuning_result = RunKernel(kernel.source(), kernel, 0, 1);
  tuning_result.status = VerifyOutput();

  if (parameter_values.size() > 0) {
    tuning_result.configuration = configuration;
  }

  // Prints the result of the tuning
  PrintHeader("Printing kernel run result to stdout");
  if (tuning_result.status) {
    PrintResult(stdout, tuning_result, kMessageResult);
  }
  else {
    PrintResult(stdout, tuning_result, kMessageWarning);
  }

  return ConvertTuningResultToPublic(tuning_result);
}

// =================================================================================================

std::vector<PublicTunerResult> TunerImpl::TuneSingleKernel(const size_t id, const bool test_reference,
                                                           const bool clear_previous_results) {
  if (clear_previous_results) {
    tuning_results_.clear();
  }

  // Runs the reference kernel if it is defined
  if (test_reference) {
    RunReferenceKernel();
  }

  KernelInfo& kernel = kernels_.at(id);
  PrintHeader("Testing kernel " + kernel.name());

  // If there are no tuning parameters, simply run the kernel and store the results
  if (kernel.parameters().size() == 0) {

    // Compiles and runs the kernel
    auto tuning_result = RunKernel(kernel.source(), kernel, 0, 1);
    tuning_result.status = VerifyOutput();

    // Stores the result of the tuning
    tuning_results_.push_back(tuning_result);

    // Else: there are tuning parameters to iterate over
  }
  else {

    // Computes the permutations of all parameters and pass them to a (smart) search algorithm
    #ifdef VERBOSE
      fprintf(stdout, "%s Computing the permutations of all parameters\n", kMessageVerbose.c_str());
    #endif

    // Creates the selected search algorithm
    std::unique_ptr<Searcher> searcher = GetSearcher(id);

    // Iterates over all possible configurations (the permutations of the tuning parameters)
    for (auto p = size_t{ 0 }; p < searcher->NumConfigurations(); ++p) {
      #ifdef VERBOSE
        fprintf(stdout, "%s Exploring configuration (%zu out of %zu)\n", kMessageVerbose.c_str(),
                p + 1, search->NumConfigurations());
      #endif
      auto permutation = searcher->GetConfiguration();

      // Adds the parameters to the source-code string as defines
      std::string source = GetConfiguredKernelSource(id, permutation);

      // Updates the local range with the parameter values
      kernel.ComputeRanges(permutation);

      // Updates number of kernel iterations based on parameter value
      kernel.SetNumCurrentIterations(permutation);

      // Compiles and runs the kernel
      auto tuning_result = RunKernel(source, kernel, p, searcher->NumConfigurations());
      tuning_result.status = VerifyOutput();

      // Gives timing feedback to the search algorithm and calculates the next index
      searcher->PushExecutionTime(tuning_result.time);
      searcher->CalculateNextIndex();

      // Stores the parameters and the timing-result
      tuning_result.configuration = permutation;
      if (tuning_result.time == std::numeric_limits<float>::max()) {
        tuning_result.time = 0.0;
        PrintResult(stdout, tuning_result, kMessageFailure);
        tuning_result.time = std::numeric_limits<float>::max();
        tuning_result.status = false;
      }
      else if (!tuning_result.status) {
        PrintResult(stdout, tuning_result, kMessageWarning);
      }
      tuning_results_.push_back(tuning_result);
    }

    // Prints a log of the searching process. This is disabled per default, but can be enabled
    // using the "OutputSearchLog" function.
    if (output_search_process_) {
      auto file = fopen(search_log_filename_.c_str(), "w");
      searcher->PrintLog(file);
      fclose(file);
    }
  }

  std::vector<PublicTunerResult> public_results;
  if(clear_previous_results) {
    for (auto &result : tuning_results_) {
      public_results.push_back(ConvertTuningResultToPublic(result));
    }
  }
  return public_results;
}

// =================================================================================================

// Starts the tuning process. First, the reference kernel is run if it exists (output results are
// automatically verified with respect to this reference run). Next, all permutations of all tuning-
// parameters are computed for each kernel and those kernels are run. Their timing-results are
// collected and stored into the tuning_results_ vector.
std::vector<PublicTunerResult> TunerImpl::TuneAllKernels() {
  // Clears tuning results from previous runs
  tuning_results_.clear();

  RunReferenceKernel();

  // Iterates over all tunable kernels
  for (size_t id = 0; id < kernels_.size(); id++) {
    std::vector<PublicTunerResult> partial_results = TuneSingleKernel(id, false, false);
  }

  std::vector<PublicTunerResult> public_results;

  for (auto &result : tuning_results_) {
    public_results.push_back(ConvertTuningResultToPublic(result));
  }

  return public_results;
}

// =================================================================================================

// Compiles the kernel and checks for error messages, sets all output buffers to zero,
// launches the kernel, and collects the timing information.
TunerImpl::TunerResult TunerImpl::RunKernel(const std::string &source, const KernelInfo &kernel,
                                            const size_t configuration_id,
                                            const size_t num_configurations) {

  // In case of an exception, skip this run
  try {

    // Compiles the kernel and prints the compiler errors/warnings
    #ifdef VERBOSE
      fprintf(stdout, "%s Starting compilation\n", kMessageVerbose.c_str());
    #endif
    auto program = Program(context_, source);
    auto options = std::vector<std::string>{};
    auto build_status = program.Build(device_, options);
    if (build_status == BuildStatus::kError) {
      auto message = program.GetBuildInfo(device_);
      fprintf(stdout, "device compiler error/warning: %s\n", message.c_str());
      throw std::runtime_error("device compiler error/warning occurred ^^\n");
    }
    if (build_status == BuildStatus::kInvalid) {
      throw std::runtime_error("Invalid program binary");
    }
    #ifdef VERBOSE
      fprintf(stdout, "%s Finished compilation\n", kMessageVerbose.c_str());
    #endif

    // Clears all previous copies of output buffer(s)
    for (auto &mem_info: arguments_output_copy_) {
      #ifdef USE_OPENCL
        CheckError(clReleaseMemObject(mem_info.buffer));
      #else
        CheckError(cuMemFree(mem_info.buffer));
      #endif
    }
    arguments_output_copy_.clear();

    // Creates a copy of the output buffer(s)
    #ifdef VERBOSE
      fprintf(stdout, "%s Creating a copy of the output buffer\n", kMessageVerbose.c_str());
    #endif
    for (auto &output : kernel.arguments_output()) {
      switch (output.type) {
        case MemType::kShort:arguments_output_copy_.push_back(CopyOutputBuffer<short>(output)); break;
        case MemType::kInt: arguments_output_copy_.push_back(CopyOutputBuffer<int>(output)); break;
        case MemType::kSizeT: arguments_output_copy_.push_back(CopyOutputBuffer<size_t>(output)); break;
        case MemType::kHalf: arguments_output_copy_.push_back(CopyOutputBuffer<half>(output)); break;
        case MemType::kFloat: arguments_output_copy_.push_back(CopyOutputBuffer<float>(output)); break;
        case MemType::kDouble: arguments_output_copy_.push_back(CopyOutputBuffer<double>(output)); break;
        case MemType::kFloat2: arguments_output_copy_.push_back(CopyOutputBuffer<float2>(output)); break;
        case MemType::kDouble2: arguments_output_copy_.push_back(CopyOutputBuffer<double2>(output)); break;
        default: throw std::runtime_error("Unsupported reference output data-type");
      }
    }

    // Sets the global and local thread-sizes
    auto global = kernel.global();
    auto local = kernel.local();

    // Runs the kernel specified number of iterations over different input / output sections
    float total_elapsed_time = 0.0f;
    for (auto iteration = size_t{ 0 }; iteration < kernel.num_current_iterations(); iteration++) {
      // Sets the kernel and its arguments
      #ifdef VERBOSE
        fprintf(stdout, "%s Setting kernel arguments\n", kMessageVerbose.c_str());
      #endif
      auto tune_kernel = Kernel(program, kernel.name());
      
      if (kernel.num_current_iterations() == 1) {
        for (auto &i : kernel.arguments_input()) { tune_kernel.SetArgument(i.index, i.buffer); }
        for (auto &i : arguments_output_copy_) { tune_kernel.SetArgument(i.index, i.buffer); }
      }
      else {
        // If kernel runs over multiple iterations, buffer is split into smaller sections of equal
        // size. Different section is used in each iteration. The buffer is split in different way
        // based on whether OpenCL or CUDA is used.
        #ifdef USE_OPENCL
          for (auto &i : kernel.arguments_input()) {
            cl_buffer_region region;
            size_t memory_per_iteration = i.size * sizeof(i.type) / kernel.num_current_iterations();
            region.origin = memory_per_iteration * iteration;
            region.size = memory_per_iteration;
            tune_kernel.SetArgument(i.index, clCreateSubBuffer(i.buffer, CL_MEM_READ_WRITE,
                                              CL_BUFFER_CREATE_TYPE_REGION, &region, NULL));
          }
          for (auto &i : arguments_output_copy_) {
            cl_buffer_region region;
            size_t memory_per_iteration = i.size * sizeof(i.type) / kernel.num_current_iterations();
            region.origin = memory_per_iteration * iteration;
            region.size = memory_per_iteration;
            tune_kernel.SetArgument(i.index, clCreateSubBuffer(i.buffer, CL_MEM_READ_WRITE,
                                              CL_BUFFER_CREATE_TYPE_REGION, &region, NULL));
          }
        #else
          // Warning: CUDA version of buffer split was NOT tested yet
          for (auto &i : kernel.arguments_input()) {
            size_t memory_per_iteration = i.size * sizeof(i.type) / kernel.num_current_iterations();
            tune_kernel.SetArgument(i.index, i.buffer + memory_per_iteration * iteration);
          }
          for (auto &i : arguments_output_copy_) {
            size_t memory_per_iteration = i.size * sizeof(i.type) / kernel.num_current_iterations();
            tune_kernel.SetArgument(i.index, i.buffer + memory_per_iteration * iteration);
          }
        #endif
      }
      for (auto &i : kernel.arguments_int()) { tune_kernel.SetArgument(i.first, i.second); }
      for (auto &i : kernel.arguments_size_t()) { tune_kernel.SetArgument(i.first, i.second); }
      for (auto &i : kernel.arguments_float()) { tune_kernel.SetArgument(i.first, i.second); }
      for (auto &i : kernel.arguments_double()) { tune_kernel.SetArgument(i.first, i.second); }
      for (auto &i : kernel.arguments_float2()) { tune_kernel.SetArgument(i.first, i.second); }
      for (auto &i : kernel.arguments_double2()) { tune_kernel.SetArgument(i.first, i.second); }

      // Verifies the local memory usage of the kernel
      auto local_mem_usage = tune_kernel.LocalMemUsage(device_);
      if (!device_.IsLocalMemoryValid(local_mem_usage)) {
          throw std::runtime_error("Using too much local memory");
      }

      // Prepares the kernel
      queue_.Finish();

      // Runs the kernel (this is the timed part)
      if (kernel.num_current_iterations() == 1) {
        fprintf(stdout, "%s Running %s\n", kMessageRun.c_str(), kernel.name().c_str());
      }
      else {
        fprintf(stdout, "%s Running %s (Iteration %u / %u)\n", kMessageRun.c_str(),
                kernel.name().c_str(), iteration + 1, kernel.num_current_iterations());
      }
      auto events = std::vector<Event>(num_runs_);
      for (auto t = size_t{ 0 }; t<num_runs_; ++t) {
        #ifdef VERBOSE
          fprintf(stdout, "%s Launching kernel (%zu out of %zu for averaging)\n",
                  kMessageVerbose.c_str(), t + 1, num_runs_);
        #endif
        tune_kernel.Launch(queue_, global, local, events[t].pointer());
        queue_.Finish(events[t]);
      }
      queue_.Finish();

      // Collects the timing information
      auto elapsed_time = std::numeric_limits<float>::max();
      for (auto t = size_t{ 0 }; t<num_runs_; ++t) {
        auto this_elapsed_time = events[t].GetElapsedTime();
        elapsed_time = std::min(elapsed_time, this_elapsed_time);
      }

      total_elapsed_time += elapsed_time;
    }

    // Prints diagnostic information
    fprintf(stdout, "%s Completed %s (%.1lf ms) - %zu out of %zu\n",
            kMessageOK.c_str(), kernel.name().c_str(), total_elapsed_time,
            configuration_id+1, num_configurations);

    // Computes the result of the tuning
    auto local_threads = size_t{ 1 };
    for (auto &item : local) { local_threads *= item; }
    TunerResult result = {kernel.name(), total_elapsed_time, local_threads, false, {}};
    return result;
  }

  // There was an exception, now return an invalid tuner results
  catch(std::exception& e) {
    fprintf(stdout, "%s Kernel %s failed\n", kMessageFailure.c_str(), kernel.name().c_str());
    fprintf(stdout, "%s   caught exception: %s\n", kMessageFailure.c_str(), e.what());
    TunerResult result = {kernel.name(), std::numeric_limits<float>::max(), 0, false, {}};
    return result;
  }
}

// =================================================================================================

// Converts TunerResult object to PublicTunerResult.
PublicTunerResult TunerImpl::ConvertTuningResultToPublic(const TunerImpl::TunerResult &result) {
  PublicTunerResult public_result;
  public_result.kernel_name = result.kernel_name;
  public_result.time = result.time;
  public_result.threads = result.threads;
  public_result.status = result.status;

  for (auto &parameter : result.configuration) {
    public_result.parameter_values.push_back(std::make_pair(parameter.name, parameter.value));
  }

  return public_result;
}

// =================================================================================================

// Returns searcher for specified kernel.
std::unique_ptr<Searcher> TunerImpl::GetSearcher(const size_t id) {
  KernelInfo& kernel = kernels_.at(id);
  kernel.SetConfigurations();

  // Creates the selected search algorithm
  std::unique_ptr<Searcher> searcher;
  switch (kernel.search_method()) {
   case SearchMethod::FullSearch:
    searcher.reset(new FullSearch{ kernel.configurations() });
    break;
   case SearchMethod::RandomSearch:
    searcher.reset(new RandomSearch{ kernel.configurations(), kernel.search_args().at(0) });
    break;
   case SearchMethod::Annealing:
    searcher.reset(new Annealing{ kernel.configurations(), kernel.search_args().at(0),
                                  kernel.search_args().at(1) });
    break;
   case SearchMethod::PSO:
    searcher.reset(new PSO{ kernel.configurations(), kernel.parameters(), kernel.search_args().at(0),
                            static_cast<size_t>(kernel.search_args().at(1)), kernel.search_args().at(2),
                            kernel.search_args().at(3), kernel.search_args().at(4) });
    break;
  }

  return searcher;
}

// =================================================================================================

// Initializes searcher of a given kernel.
void TunerImpl::InitializeSearcher(const size_t id) {
  KernelInfo& kernel = kernels_.at(id);
  kernel.SetConfigurations();

  switch (kernel.search_method()) {
   case SearchMethod::FullSearch:
    kernel_searchers_.at(id).reset(new FullSearch{ kernel.configurations() });
    break;
   case SearchMethod::RandomSearch:
    kernel_searchers_.at(id).reset(new RandomSearch{ kernel.configurations(), kernel.search_args().at(0) });
    break;
   case SearchMethod::Annealing:
    kernel_searchers_.at(id).reset(new Annealing{ kernel.configurations(), kernel.search_args().at(0),
                                                  kernel.search_args().at(1) });
    break;
   case SearchMethod::PSO:
    kernel_searchers_.at(id).reset(new PSO{ kernel.configurations(), kernel.parameters(), kernel.search_args().at(0),
                                            static_cast<size_t>(kernel.search_args().at(1)), kernel.search_args().at(2),
                                            kernel.search_args().at(3), kernel.search_args().at(4) });
    break;
  }
}

// =================================================================================================

// Returns number of unique configurations for given kernel.
size_t TunerImpl::GetNumConfigurations(const size_t id) {
  if(kernel_searchers_.at(id) == nullptr) {
    InitializeSearcher(id);
  }

  return kernel_searchers_.at(id)->NumConfigurations();
}

// =================================================================================================

// Returns next configuration for given kernel.
KernelInfo::Configuration TunerImpl::GetNextConfiguration(const size_t id) const {
  if(kernel_searchers_.at(id) == nullptr) {
    throw std::runtime_error("Next configuration might not exist, call GetNumConfigurations() method first.");
  }

  return kernel_searchers_.at(id)->GetConfiguration();
}

// =================================================================================================

// Updates searcher with given info.
void TunerImpl::UpdateSearcher(const size_t id, const float previous_running_time) {
  if(kernel_searchers_.at(id) == nullptr) {
    throw std::runtime_error("Searcher for given kernel is not initialized.");
  }

  kernel_searchers_.at(id)->PushExecutionTime(previous_running_time);
  kernel_searchers_.at(id)->CalculateNextIndex();
}

// =================================================================================================

// Returns modified kernel source (with #defines) based on provided configuration.
std::string TunerImpl::GetConfiguredKernelSource(const size_t id,
                                                 const KernelInfo::Configuration& configuration) {
  auto source = std::string{};
  for (auto &config : configuration) {
    source += config.GetDefine();
  }
  source += kernels_.at(id).source();

  return source;
}

// =================================================================================================

// Runs reference kernel and stores its result.
void TunerImpl::RunReferenceKernel() {
  if (has_reference_) {
    PrintHeader("Testing reference " + reference_kernel_->name());
    RunKernel(reference_kernel_->source(), *reference_kernel_, 0, 1);
    StoreReferenceOutput();
  }
}

// =================================================================================================

// Uploads a copy of the output vector to the device. This is done because the output might as well
// be an input buffer at the same time. Every kernel might override it, so it needs to be updated
// before each run.
template <typename T>
KernelInfo::MemArgument TunerImpl::CopyOutputBuffer(KernelInfo::MemArgument &argument) {
  auto buffer_copy = Buffer<T>(context_, BufferAccess::kNotOwned, argument.size);
  auto buffer_source = Buffer<T>(argument.buffer);
  buffer_source.CopyTo(queue_, argument.size, buffer_copy);
  auto result = KernelInfo::MemArgument{argument.index, argument.size, argument.type, buffer_copy()};
  return result;
}

// =================================================================================================

// Loops over all reference outputs, creates per output a new host buffer and copies the device
// buffer from the device onto the host. This function is specialised for different data-types.
void TunerImpl::StoreReferenceOutput() {
  reference_outputs_.clear();
  for (auto &output_buffer: arguments_output_copy_) {
    switch (output_buffer.type) {
      case MemType::kShort: DownloadReference<short>(output_buffer); break;
      case MemType::kInt: DownloadReference<int>(output_buffer); break;
      case MemType::kSizeT: DownloadReference<size_t>(output_buffer); break;
      case MemType::kHalf: DownloadReference<half>(output_buffer); break;
      case MemType::kFloat: DownloadReference<float>(output_buffer); break;
      case MemType::kDouble: DownloadReference<double>(output_buffer); break;
      case MemType::kFloat2: DownloadReference<float2>(output_buffer); break;
      case MemType::kDouble2: DownloadReference<double2>(output_buffer); break;
      default: throw std::runtime_error("Unsupported reference output data-type");
    }
  }
}
template <typename T> void TunerImpl::DownloadReference(KernelInfo::MemArgument &device_buffer) {
  auto host_buffer = new T[device_buffer.size];
  Buffer<T>(device_buffer.buffer).Read(queue_, device_buffer.size, host_buffer);
  reference_outputs_.push_back(host_buffer);
}

// =================================================================================================

// In case there is a reference kernel, this function loops over all outputs, creates per output a
// new host buffer and copies the device buffer from the device onto the host. Following, it
// compares the results to the reference output. This function is specialised for different
// data-types. These functions return "true" if everything is OK, and "false" if there is a warning.
bool TunerImpl::VerifyOutput() {
  auto status = true;
  if (has_reference_) {
    auto i = size_t{0};
    for (auto &output_buffer: arguments_output_copy_) {
      switch (output_buffer.type) {
        case MemType::kShort: status &= DownloadAndCompare<short>(output_buffer, i); break;
        case MemType::kInt: status &= DownloadAndCompare<int>(output_buffer, i); break;
        case MemType::kSizeT: status &= DownloadAndCompare<size_t>(output_buffer, i); break;
        case MemType::kHalf: status &= DownloadAndCompare<half>(output_buffer, i); break;
        case MemType::kFloat: status &= DownloadAndCompare<float>(output_buffer, i); break;
        case MemType::kDouble: status &= DownloadAndCompare<double>(output_buffer, i); break;
        case MemType::kFloat2: status &= DownloadAndCompare<float2>(output_buffer, i); break;
        case MemType::kDouble2: status &= DownloadAndCompare<double2>(output_buffer, i); break;
        default: throw std::runtime_error("Unsupported output data-type");
      }
      ++i;
    }
  }
  return status;
}

// See above comment
template <typename T>
bool TunerImpl::DownloadAndCompare(KernelInfo::MemArgument &device_buffer, const size_t i) {
  auto l2_norm = 0.0;

  // Downloads the results to the host
  std::vector<T> host_buffer(device_buffer.size);
  Buffer<T>(device_buffer.buffer).Read(queue_, device_buffer.size, host_buffer);
  T* reference_output = static_cast<T*>(reference_outputs_[i]);

  // Compares the results (L2 norm)
  if (verification_method_ == VerificationMethod::AbsoluteDifference) {
    for (auto j = size_t{ 0 }; j<device_buffer.size; ++j) {
      l2_norm += AbsoluteDifference(reference_output[j], host_buffer[j]);
    }

    // Verifies if everything was OK, if not: print the L2 norm
    if (std::isnan(l2_norm) || l2_norm > tolerance_treshold_) {
      fprintf(stderr, "%s Results differ: L2 norm is %6.2e\n", kMessageWarning.c_str(), l2_norm);
      return false;
    }
    return true;
  }
  else if (verification_method_ == VerificationMethod::SideBySide) {
    for (auto j = size_t{ 0 }; j<device_buffer.size; ++j)
    {
      if (AbsoluteDifference(reference_output[j], host_buffer[j]) > tolerance_treshold_) {
        fprintf(stderr, "%s Different results for position %u in output: difference is %.8lf\n",
                kMessageWarning.c_str(), j, AbsoluteDifference(reference_output[j], host_buffer[j]));
        return false;
      }
    }
    return true;
  }
}

// Computes the absolute difference
template <typename T>
double TunerImpl::AbsoluteDifference(const T reference, const T result) {
  return fabs(static_cast<double>(reference) - static_cast<double>(result));
}
template <> double TunerImpl::AbsoluteDifference(const float2 reference, const float2 result) {
  auto real = fabs(static_cast<double>(reference.real()) - static_cast<double>(result.real()));
  auto imag = fabs(static_cast<double>(reference.imag()) - static_cast<double>(result.imag()));
  return real + imag;
}
template <> double TunerImpl::AbsoluteDifference(const double2 reference, const double2 result) {
  auto real = fabs(reference.real() - result.real());
  auto imag = fabs(reference.imag() - result.imag());
  return real + imag;
}
template <> double TunerImpl::AbsoluteDifference(const half reference, const half result) {
  const auto reference_float = HalfToFloat(reference);
  const auto result_float = HalfToFloat(result);
  return fabs(static_cast<double>(reference_float) - static_cast<double>(result_float));
}

// =================================================================================================

// Trains a model and predicts all remaining configurations
void TunerImpl::ModelPrediction(const Model model_type, const float validation_fraction,
                                const size_t test_top_x_configurations) {

  // Iterates over all tunable kernels
  for (auto &kernel: kernels_) {

    // Retrieves the number of training samples and features
    auto validation_samples = static_cast<size_t>(tuning_results_.size()*validation_fraction);
    auto training_samples = tuning_results_.size() - validation_samples;
    auto features = tuning_results_[0].configuration.size();

    // Sets the raw training and validation data
    auto x_train = std::vector<std::vector<float>>(training_samples, std::vector<float>(features));
    auto y_train = std::vector<float>(training_samples);
    for (auto s=size_t{0}; s<training_samples; ++s) {
      y_train[s] = tuning_results_[s].time;
      for (auto f=size_t{0}; f<features; ++f) {
        x_train[s][f] = static_cast<float>(tuning_results_[s].configuration[f].value);
      }
    }
    auto x_validation = std::vector<std::vector<float>>(validation_samples, std::vector<float>(features));
    auto y_validation = std::vector<float>(validation_samples);
    for (auto s=size_t{0}; s<validation_samples; ++s) {
      y_validation[s] = tuning_results_[s+training_samples].time;
      for (auto f=size_t{0}; f<features; ++f) {
        x_validation[s][f] = static_cast<float>(tuning_results_[s + training_samples].configuration[f].value);
      }
    }

    // Pointer to one of the machine learning models
    std::unique_ptr<MLModel<float>> model;

    // Trains a linear regression model
    if (model_type == Model::kLinearRegression) {
      PrintHeader("Training a linear regression model");

      // Sets the learning parameters
      auto learning_iterations = size_t{800}; // For gradient descent
      auto learning_rate = 0.05f; // For gradient descent
      auto lambda = 0.2f; // Regularization parameter
      auto debug_display = true; // Output learned data to stdout

      // Trains and validates the model
      model = std::unique_ptr<MLModel<float>>(
        new LinearRegression<float>(learning_iterations, learning_rate, lambda, debug_display)
      );
      model->Train(x_train, y_train);
      model->Validate(x_validation, y_validation);
    }

    // Trains a neural network model
    else if (model_type == Model::kNeuralNetwork) {
      PrintHeader("Training a neural network model");

      // Sets the learning parameters
      auto learning_iterations = size_t{800}; // For gradient descent
      auto learning_rate = 0.1f; // For gradient descent
      auto lambda = 0.005f; // Regularization parameter
      auto debug_display = true; // Output learned data to stdout
      auto layers = std::vector<size_t>{features, 20, 1};

      // Trains and validates the model
      model = std::unique_ptr<MLModel<float>>(
        new NeuralNetwork<float>(learning_iterations, learning_rate, lambda, layers, debug_display)
      );
      model->Train(x_train, y_train);
      model->Validate(x_validation, y_validation);
    }

    // Unknown model
    else {
      throw std::runtime_error("Unknown machine learning model");
    }

    // Iterates over all configurations (the permutations of the tuning parameters)
    PrintHeader("Predicting the remaining configurations using the model");
    auto model_results = std::vector<std::tuple<size_t,float>>();
    auto p = size_t{0};
    for (auto &permutation: kernel.configurations()) {

      // Runs the trained model to predicts the result
      auto x_test = std::vector<float>();
      for (auto &setting: permutation) {
        x_test.push_back(static_cast<float>(setting.value));
      }
      auto predicted_time = model->Predict(x_test);
      model_results.push_back(std::make_tuple(p, predicted_time));
      ++p;
    }

    // Sorts the modelled results by performance
    std::sort(begin(model_results), end(model_results),
      [](const std::tuple<size_t,float> &t1, const std::tuple<size_t,float> &t2) {
        return std::get<1>(t1) < std::get<1>(t2);
      }
    );

    // Tests the best configurations on the device to verify the results
    PrintHeader("Testing the best-found configurations");
    for (auto i=size_t{0}; i<test_top_x_configurations && i<model_results.size(); ++i) {
      auto result = model_results[i];
      printf("[ -------> ] The model predicted: %.3lf ms\n", std::get<1>(result));
      auto pid = std::get<0>(result);
      auto permutations = kernel.configurations();
      auto permutation = permutations[pid];

      // Adds the parameters to the source-code string as defines
      auto source = std::string{};
      for (auto &config: permutation) {
        source += config.GetDefine();
      }
      source += kernel.source();

      // Updates the local range with the parameter values
      kernel.ComputeRanges(permutation);

      // Compiles and runs the kernel
      auto tuning_result = RunKernel(source, kernel, pid, test_top_x_configurations);
      tuning_result.status = VerifyOutput();

      // Stores the parameters and the timing-result
      tuning_result.configuration = permutation;
      tuning_results_.push_back(tuning_result);
      if (tuning_result.time == std::numeric_limits<float>::max()) {
        tuning_result.time = 0.0;
        PrintResult(stdout, tuning_result, kMessageFailure);
        tuning_result.time = std::numeric_limits<float>::max();
      }
      else if (!tuning_result.status) {
        PrintResult(stdout, tuning_result, kMessageWarning);
      }
    }
  }
}

// =================================================================================================

// Prints a result by looping over all its configuration parameters
void TunerImpl::PrintResult(FILE* fp, const TunerResult &result, const std::string &message) const {
  fprintf(fp, "%s %s; ", message.c_str(), result.kernel_name.c_str());
  fprintf(fp, "%8.1lf ms;", result.time);
  for (auto &setting: result.configuration) {
    fprintf(fp, "%9s;", setting.GetConfig().c_str());
  }
  fprintf(fp, "\n");
}

// =================================================================================================

// Loads a file into a stringstream and returns the result as a string
std::string TunerImpl::LoadFile(const std::string &filename) {
  std::ifstream file(filename);
  if (file.fail()) { throw std::runtime_error("Could not open kernel file: "+filename); }
  std::stringstream file_contents;
  file_contents << file.rdbuf();
  return file_contents.str();
}

// =================================================================================================

// Converts a C++ string to a C string and print it out with nice formatting
void TunerImpl::PrintHeader(const std::string &header_name) const {
  if (!suppress_output_) {
    fprintf(stdout, "\n%s %s\n", kMessageHead.c_str(), header_name.c_str());
  }
}

// =================================================================================================

// Get the MemType based on a template argument
template <> MemType TunerImpl::GetType<short>() { return MemType::kShort; }
template <> MemType TunerImpl::GetType<int>() { return MemType::kInt; }
template <> MemType TunerImpl::GetType<size_t>() { return MemType::kSizeT; }
template <> MemType TunerImpl::GetType<half>() { return MemType::kHalf; }
template <> MemType TunerImpl::GetType<float>() { return MemType::kFloat; }
template <> MemType TunerImpl::GetType<double>() { return MemType::kDouble; }
template <> MemType TunerImpl::GetType<float2>() { return MemType::kFloat2; }
template <> MemType TunerImpl::GetType<double2>() { return MemType::kDouble2; }

// =================================================================================================
} // namespace cltune
