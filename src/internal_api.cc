
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the Tuner class (see the header for information about the class).
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
#include "internal/internal_api.h"

// And the implemenation (Pimpl idiom)
#include "internal/tuner_impl.h"

#include <iostream> // FILE
#include <limits> // std::numeric_limits

namespace cltune {
// =================================================================================================

// The implemenation of the constructors and destructors are hidden in the TunerImpl class
Tuner::Tuner():
    pimpl(new TunerImpl()) {
}
Tuner::Tuner(size_t platform_id, size_t device_id):
    pimpl(new TunerImpl(platform_id, device_id)) {
}
Tuner::~Tuner() {
}

// =================================================================================================

// Loads the kernel source-code from a file and calls the function-overload below.
size_t Tuner::AddKernel(const std::vector<std::string> &filenames, const std::string &kernel_name,
                        const IntRange &global, const IntRange &local) {
  auto source = std::string{};
  for (auto &filename: filenames) {
    source += pimpl->LoadFile(filename);
  }
  return AddKernelFromString(source, kernel_name, global, local);
}

// Loads the kernel source-code from a string and creates a new variable of type KernelInfo to store
// all the kernel-information.
size_t Tuner::AddKernelFromString(const std::string &source, const std::string &kernel_name,
                                  const IntRange &global, const IntRange &local) {
  pimpl->kernels_.push_back(KernelInfo(kernel_name, source, pimpl->device()));
  pimpl->kernel_searchers_.push_back(nullptr);
  auto id = pimpl->kernels_.size() - 1;
  pimpl->kernels_[id].set_global_base(global);
  pimpl->kernels_[id].set_local_base(local);
  return id;
}

// =================================================================================================

// Sets the reference kernel (source-code location, kernel name, global/local thread-sizes) and
// sets a flag to indicate that there is now a reference. Calling this function again will simply
// overwrite the old reference.
void Tuner::SetReference(const std::vector<std::string> &filenames, const std::string &kernel_name,
                         const IntRange &global, const IntRange &local) {
  auto source = std::string{};
  for (auto &filename: filenames) {
    source += pimpl->LoadFile(filename);
  }
  SetReferenceFromString(source, kernel_name, global, local);
}
void Tuner::SetReferenceFromString(const std::string &source, const std::string &kernel_name,
                                   const IntRange &global, const IntRange &local) {
  pimpl->has_reference_ = true;
  pimpl->reference_kernel_.reset(new KernelInfo(kernel_name, source, pimpl->device()));
  pimpl->reference_kernel_->set_global_base(global);
  pimpl->reference_kernel_->set_local_base(local);
}

// =================================================================================================

// Adds parameters for a kernel to tune. Also checks whether this parameter already exists.
void Tuner::AddParameter(const size_t id, const std::string &parameter_name,
                         const std::initializer_list<size_t> &values) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  if (pimpl->kernels_[id].ParameterExists(parameter_name)) {
    throw std::runtime_error("Parameter already exists");
  }
  pimpl->kernels_[id].AddParameter(parameter_name, values);
}

// As above, but now adds a single valued parameter to the reference
void Tuner::AddParameterReference(const std::string &parameter_name, const size_t value) {
  auto value_string = std::string{std::to_string(static_cast<long long>(value))};
  pimpl->reference_kernel_->PrependSource("#define "+parameter_name+" "+value_string);
}

// =================================================================================================

// These functions forward their work (adding a modifier to global/local thread-sizes) to an object
// of KernelInfo class
void Tuner::MulGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalMul);
}
void Tuner::DivGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalDiv);
}
void Tuner::AddGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalAdd);
}
void Tuner::MulLocalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalMul);
}
void Tuner::DivLocalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalDiv);
}

void Tuner::SetMultirunKernelIterations(const size_t id, const std::string &parameter_name) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  if (!pimpl->kernels_[id].ParameterExists(parameter_name)) {
    throw std::runtime_error("Invalid parameter name");
  }
  
  for (auto parameter : pimpl->kernels_[id].parameters()) {
    if (parameter.name == parameter_name) {
        for (auto value : parameter.values) {
          if (value < 1) { throw std::runtime_error("Invalid number of iterations"); }
        }
        pimpl->kernels_[id].set_iterations(parameter.values, parameter.name);
    }
  }
}

// Adds a contraint to the list of constraints for a particular kernel. First checks whether the
// kernel exists and whether the parameters exist.
void Tuner::AddConstraint(const size_t id, ConstraintFunction valid_if,
                          const std::vector<std::string> &parameters) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!pimpl->kernels_[id].ParameterExists(parameter)) {
      throw std::runtime_error("Invalid parameter");
    }
  }
  pimpl->kernels_[id].AddConstraint(valid_if, parameters);
}

// As above, but for the local memory usage
void Tuner::SetLocalMemoryUsage(const size_t id, LocalMemoryFunction amount,
                                const std::vector<std::string> &parameters) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!pimpl->kernels_[id].ParameterExists(parameter)) {
      throw std::runtime_error("Invalid parameter");
    }
  }
  pimpl->kernels_[id].SetLocalMemoryUsage(amount, parameters);
}


// =================================================================================================

// Creates a new buffer of type Memory (containing both host and device data) based on a source
// vector of data. Then, upload it to the device and store the argument in a list.
template <typename T>
void Tuner::AddArgumentInput(const size_t id, const std::vector<T> &source) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  auto device_buffer = Buffer<T>(pimpl->context(), BufferAccess::kNotOwned, source.size());
  device_buffer.Write(pimpl->queue(), source.size(), source);
  auto argument = KernelInfo::MemArgument{ pimpl->kernels_[id].argument_counter(), source.size(),
                                         pimpl->GetType<T>(), device_buffer()};
  pimpl->kernels_[id].AddArgumentInput(argument);
}

// Compiles the function for various data-types
template void PUBLIC_API Tuner::AddArgumentInput<short>(const size_t id, const std::vector<short>&);
template void PUBLIC_API Tuner::AddArgumentInput<int>(const size_t id, const std::vector<int>&);
template void PUBLIC_API Tuner::AddArgumentInput<size_t>(const size_t id, const std::vector<size_t>&);
template void PUBLIC_API Tuner::AddArgumentInput<half>(const size_t id, const std::vector<half>&);
template void PUBLIC_API Tuner::AddArgumentInput<float>(const size_t id, const std::vector<float>&);
template void PUBLIC_API Tuner::AddArgumentInput<double>(const size_t id, const std::vector<double>&);
template void PUBLIC_API Tuner::AddArgumentInput<float2>(const size_t id, const std::vector<float2>&);
template void PUBLIC_API Tuner::AddArgumentInput<double2>(const size_t id, const std::vector<double2>&);

// Same as above for reference kernel
template <typename T>
void Tuner::AddArgumentInputReference(const std::vector<T> &source) {
    auto device_buffer = Buffer<T>(pimpl->context(), BufferAccess::kNotOwned, source.size());
    device_buffer.Write(pimpl->queue(), source.size(), source);
    auto argument = KernelInfo::MemArgument{ pimpl->reference_kernel_->argument_counter(), source.size(),
        pimpl->GetType<T>(), device_buffer() };
    pimpl->reference_kernel_->AddArgumentInput(argument);
}

template void PUBLIC_API Tuner::AddArgumentInputReference<short>(const std::vector<short>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<int>(const std::vector<int>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<size_t>(const std::vector<size_t>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<half>(const std::vector<half>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<float>(const std::vector<float>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<double>(const std::vector<double>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<float2>(const std::vector<float2>&);
template void PUBLIC_API Tuner::AddArgumentInputReference<double2>(const std::vector<double2>&);

// Similar to the above function, but now marked as output buffer. Output buffers are special in the
// sense that they will be checked in the verification process.
template <typename T>
void Tuner::AddArgumentOutput(const size_t id, const std::vector<T> &source) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  auto device_buffer = Buffer<T>(pimpl->context(), BufferAccess::kNotOwned, source.size());
  device_buffer.Write(pimpl->queue(), source.size(), source);
  auto argument = KernelInfo::MemArgument{ pimpl->kernels_[id].argument_counter(), source.size(),
                                         pimpl->GetType<T>(), device_buffer()};
  pimpl->kernels_[id].AddArgumentOutput(argument);
}

// Compiles the function for various data-types
template void PUBLIC_API Tuner::AddArgumentOutput<short>(const size_t id, const std::vector<short>&);
template void PUBLIC_API Tuner::AddArgumentOutput<int>(const size_t id, const std::vector<int>&);
template void PUBLIC_API Tuner::AddArgumentOutput<size_t>(const size_t id, const std::vector<size_t>&);
template void PUBLIC_API Tuner::AddArgumentOutput<half>(const size_t id, const std::vector<half>&);
template void PUBLIC_API Tuner::AddArgumentOutput<float>(const size_t id, const std::vector<float>&);
template void PUBLIC_API Tuner::AddArgumentOutput<double>(const size_t id, const std::vector<double>&);
template void PUBLIC_API Tuner::AddArgumentOutput<float2>(const size_t id, const std::vector<float2>&);
template void PUBLIC_API Tuner::AddArgumentOutput<double2>(const size_t id, const std::vector<double2>&);

// Same as above for reference kernel
template <typename T>
void Tuner::AddArgumentOutputReference(const std::vector<T> &source) {
    auto device_buffer = Buffer<T>(pimpl->context(), BufferAccess::kNotOwned, source.size());
    device_buffer.Write(pimpl->queue(), source.size(), source);
    auto argument = KernelInfo::MemArgument{ pimpl->reference_kernel_->argument_counter(), source.size(),
        pimpl->GetType<T>(), device_buffer() };
    pimpl->reference_kernel_->AddArgumentOutput(argument);
}

template void PUBLIC_API Tuner::AddArgumentOutputReference<short>(const std::vector<short>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<int>(const std::vector<int>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<size_t>(const std::vector<size_t>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<half>(const std::vector<half>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<float>(const std::vector<float>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<double>(const std::vector<double>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<float2>(const std::vector<float2>&);
template void PUBLIC_API Tuner::AddArgumentOutputReference<double2>(const std::vector<double2>&);

// Sets a scalar value as an argument to the kernel. Since a vector of scalars of any type doesn't
// exist, there is no general implemenation. Instead, each data-type has its specialised version in
// which it stores to a specific vector.
template <> void PUBLIC_API Tuner::AddArgumentScalar<short>(const size_t id, const short argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<int>(const size_t id, const int argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<size_t>(const size_t id, const size_t argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<half>(const size_t id, const half argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<float>(const size_t id, const float argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<double>(const size_t id, const double argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<float2>(const size_t id, const float2 argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalar<double2>(const size_t id, const double2 argument) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::ModifyArgumentScalar<int>(const size_t id, const int argument, const int index) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].ModifyArgumentScalar(argument, index);
}

// Same as above for reference kernel
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<short>(const short argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<int>(const int argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<size_t>(const size_t argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<half>(const half argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<float>(const float argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<double>(const double argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<float2>(const float2 argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}
template <> void PUBLIC_API Tuner::AddArgumentScalarReference<double2>(const double2 argument) {
    pimpl->reference_kernel_->AddArgumentScalar(argument);
}

// =================================================================================================

// Use full search as a search strategy. This is the default method.
void Tuner::UseFullSearch(const size_t id) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].UseFullSearch();
}

// Use random search as a search strategy.
void Tuner::UseRandomSearch(const size_t id, const double fraction) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].UseRandomSearch(fraction);
}

// Use simulated annealing as a search strategy.
void Tuner::UseAnnealing(const size_t id, const double fraction, const double max_temperature) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].UseAnnealing(fraction, max_temperature);
}

// Use PSO as a search strategy.
void Tuner::UsePSO(const size_t id, const double fraction, const size_t swarm_size,
                   const double influence_global, const double influence_local,
                   const double influence_random) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].UsePSO(fraction, swarm_size, influence_global, influence_local, influence_random);
}

// Choose verification method.
void Tuner::ChooseVerificationMethod(const VerificationMethod method,
                                     const double tolerance_treshold) {
  if (tolerance_treshold < 0.0) { throw std::runtime_error("Invalid tolerance treshold"); }
  pimpl->verification_method_ = method;
  pimpl->tolerance_treshold_ = tolerance_treshold;
}

// Output the search process to a file. This is disabled per default.
void Tuner::OutputSearchLog(const std::string &filename) {
  pimpl->output_search_process_ = true;
  pimpl->search_log_filename_ = filename;
}

// =================================================================================================

// Starts the tuning process. See the TunerImpl's implemenation for details
std::vector<PublicTunerResult> Tuner::TuneAllKernels() {
  return pimpl->TuneAllKernels();
}

// =================================================================================================

// Starts the tuning process for single kernel.
std::vector<PublicTunerResult> Tuner::TuneSingleKernel(const size_t id) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  return pimpl->TuneSingleKernel(id, true, true);
}

// =================================================================================================

// Modifies global / local range size for given kernel
void Tuner::ModifyGlobalRange(const size_t id, const IntRange &new_global) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].set_global_base(new_global);
}

void Tuner::ModifyLocalRange(const size_t id, const IntRange &new_local) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].set_local_base(new_local);
}

// Returns global / local range size for given kernel
IntRange Tuner::GetGlobalRange(const size_t id) const {
    if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
    return pimpl->kernels_[id].global_base();
}

IntRange Tuner::GetLocalRange(const size_t id) const {
    if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
    return pimpl->kernels_[id].local_base();
}

// =================================================================================================

// Returns number of unique configurations for given kernel based on specified parameters and search method.
size_t Tuner::GetNumConfigurations(const size_t id) const {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  return pimpl->GetNumConfigurations(id);
}

// Returns next configuration for given kernel based on search method.
ParameterRange Tuner::GetNextConfiguration(const size_t id) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }

  ParameterRange parameters;
  KernelInfo::Configuration config = pimpl->GetNextConfiguration(id);

  for (auto& config_unit : config) {
    parameters.push_back(std::make_pair(config_unit.name, config_unit.value));
  }

  return parameters;
}

// This methods needs to be called after each getNextConfiguration() method, previous kernel running time
// should be provided.
void Tuner::UpdateKernelConfiguration(const size_t id, const float previous_running_time) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->UpdateSearcher(id, previous_running_time);
}

// =================================================================================================

// Runs reference kernel and stores its result.
void Tuner::RunReferenceKernel() {
  pimpl->RunReferenceKernel();
}

// =================================================================================================

// Runs single kernel with given configuration and measures time.
PublicTunerResult Tuner::RunSingleKernel(const size_t id, const ParameterRange &parameter_values) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  return pimpl->RunSingleKernel(id, parameter_values);
}

// =================================================================================================

// Fits a machine learning model. See the TunerImpl's implemenation for details
void Tuner::ModelPrediction(const Model model_type, const float validation_fraction,
                            const size_t test_top_x_configurations) {
  pimpl->ModelPrediction(model_type, validation_fraction, test_top_x_configurations);
}

// =================================================================================================

// Iterates over all tuning results and prints each parameter configuration and the corresponding
// timing-results. Printing is to stdout.
double Tuner::PrintToScreen() const {

  // Finds the best result
  auto best_result = pimpl->tuning_results_[0];
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Aborts if there was no best time found
  if (best_time == std::numeric_limits<double>::max()) {
    pimpl->PrintHeader("No tuner results found");
    return 0.0;
  }

  // Prints all valid results and the one with the lowest execution time
  pimpl->PrintHeader("Printing results to stdout");
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && tuning_result.time != std::numeric_limits<double>::max()) {
      pimpl->PrintResult(stdout, tuning_result, pimpl->kMessageResult);
    }
  }
  pimpl->PrintHeader("Printing best result to stdout");
  pimpl->PrintResult(stdout, best_result, pimpl->kMessageBest);

  // Return the best time
  return best_time;
}

// Prints the best result in a neatly formatted C++ database format to screen
void Tuner::PrintFormatted() const {

  // Finds the best result
  auto best_result = pimpl->tuning_results_[0];
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Prints the best result in C++ database format
  auto count = size_t{0};
  pimpl->PrintHeader("Printing best result in database format to stdout");
  fprintf(stdout, "{ \"%s\", { ", pimpl->device().Name().c_str());
  for (auto &setting: best_result.configuration) {
    fprintf(stdout, "%s", setting.GetDatabase().c_str());
    if (count < best_result.configuration.size()-1) {
      fprintf(stdout, ", ");
    }
    count++;
  }
  fprintf(stdout, " } }\n");
}

// Outputs all results in a JSON database format
void Tuner::PrintJSON(const std::string &filename,
                      const std::vector<std::pair<std::string,std::string>> &descriptions) const {

  // Prints the best result in JSON database format
  pimpl->PrintHeader("Printing results to file in JSON format");
  auto file = fopen(filename.c_str(), "w");
  auto device_type = pimpl->device().Type();
  fprintf(file, "{\n");
  for (auto &description: descriptions) {
    fprintf(file, "  \"%s\": \"%s\",\n", description.first.c_str(), description.second.c_str());
  }
  fprintf(file, "  \"device\": \"%s\",\n", pimpl->device().Name().c_str());
  fprintf(file, "  \"device_vendor\": \"%s\",\n", pimpl->device().Vendor().c_str());
  fprintf(file, "  \"device_type\": \"%s\",\n", device_type.c_str());
  fprintf(file, "  \"device_core_clock\": \"%zu\",\n", pimpl->device().CoreClock());
  fprintf(file, "  \"device_compute_units\": \"%zu\",\n", pimpl->device().ComputeUnits());
  fprintf(file, "  \"results\": [\n");

  // Filters failed configurations
  auto results = std::vector<TunerImpl::TunerResult>();
  for (const auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && tuning_result.time != std::numeric_limits<double>::max()) {
      results.push_back(tuning_result);
    }
  }

  // Loops over all the results
  auto num_results = results.size();
  for (auto r=size_t{0}; r<num_results; ++r) {
    auto result = results[r];
    fprintf(file, "    {\n");
    fprintf(file, "      \"kernel\": \"%s\",\n", result.kernel_name.c_str());
    fprintf(file, "      \"time\": %.3lf,\n", result.time);

    // Loops over all the parameters for this result
    fprintf(file, "      \"parameters\": {");
    auto num_configs = result.configuration.size();
    for (auto p=size_t{0}; p<num_configs; ++p) {
      auto config = result.configuration[p];
      fprintf(file, "\"%s\": %zu", config.name.c_str(), config.value);
      if (p < num_configs-1) { fprintf(file, ","); }
    }
    fprintf(file, "}\n");

    // The footer
    fprintf(file, "    }");
    if (r < num_results-1) { fprintf(file, ","); }
    fprintf(file, "\n");
  }
  fprintf(file, "  ]\n");
  fprintf(file, "}\n");
  fclose(file);
}

// Same as PrintToScreen, but now outputs into a file and does not mark the best-case
void Tuner::PrintToFile(const std::string &filename) const {
  pimpl->PrintHeader("Printing results to file: "+filename);
  auto file = fopen(filename.c_str(), "w");
  std::vector<std::string> processed_kernels;
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status) {

      // Checks whether this is a kernel which hasn't been encountered yet
      auto new_kernel = true;
      for (auto &kernel_name: processed_kernels) {
        if (kernel_name == tuning_result.kernel_name) { new_kernel = false; break; }
      }
      processed_kernels.push_back(tuning_result.kernel_name);

      // Prints the header in case of a new kernel name
      if (new_kernel) {
        fprintf(file, "name;time;threads;");
        for (auto &setting: tuning_result.configuration) {
          fprintf(file, "%s;", setting.name.c_str());
        }
        fprintf(file, "\n");
      }

      // Prints an entry to file
      fprintf(file, "%s;", tuning_result.kernel_name.c_str());
      fprintf(file, "%.2lf;", tuning_result.time);
      fprintf(file, "%zu;", tuning_result.threads);
      for (auto &setting: tuning_result.configuration) {
        fprintf(file, "%zu;", setting.value);
      }
      fprintf(file, "\n");
    }
  }
  fclose(file);
}

// Set the flag to suppress output to true. Note that this cannot be undone.
void Tuner::SuppressOutput() {
  pimpl->suppress_output_ = true;
}

// =================================================================================================
} // namespace cltune
