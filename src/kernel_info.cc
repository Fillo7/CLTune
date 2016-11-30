
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the KernelInfo class (see the header for information about the class).
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
#include "internal/kernel_info.h"
#include "internal/searchers/full_search.h"
#include "internal/searchers/random_search.h"
#include "internal/searchers/annealing.h"
#include "internal/searchers/pso.h"

#include <cassert>

namespace cltune {
// =================================================================================================

// Initializes the name and kernel source-code, creates empty containers for all other member
// variables.
KernelInfo::KernelInfo(const std::string name, const std::string source, const Device &device):
  name_(name),
  source_(source),
  parameters_(),
  configurations_(),
  constraints_(),
  local_memory_(LocalMemory{[] (std::vector<size_t>) { return size_t{0}; }, std::vector<std::string>(0)}),
  device_(device),
  global_base_(), local_base_(),
  global_(), local_(),
  iterations_(IterationsModifier{ std::vector<size_t>{1}, std::string{""} }),
  num_current_iterations_(1),
  searcher_(nullptr),
  argument_counter_(0),
  thread_size_modifiers_() {
}

KernelInfo::~KernelInfo() {
  // Frees the device buffers
  auto free_buffers = [](MemArgument &mem_info) {
  #ifdef USE_OPENCL
    CheckError(clReleaseMemObject(mem_info.buffer));
  #else
    CheckError(cuMemFree(mem_info.buffer));
  #endif
  };
  for (auto &mem_argument : arguments_input_) { free_buffers(mem_argument); }
  for (auto &mem_argument : arguments_output_) { free_buffers(mem_argument); }
}

// =================================================================================================

void KernelInfo::PrependSource(const std::string &extra_source) {
  source_ = extra_source + "\n" + source_;
}

// =================================================================================================

// Pushes a new parameter to the list of parameters
void KernelInfo::AddParameter(const std::string &name, const std::vector<size_t> &values) {
  Parameter parameter = {name, values};
  parameters_.push_back(parameter);
}

// Loops over all parameters and checks whether the given parameter name is present
bool KernelInfo::ParameterExists(const std::string parameter_name) {
  for (auto &parameter: parameters_) {
    if (parameter.name == parameter_name) { return true; }
  }
  return false;
}

// =================================================================================================

// Pushes a new item onto the list of modifiers of a particular type
void KernelInfo::AddModifier(const StringRange range, const ThreadSizeModifierType type) {
  ThreadSizeModifier modifier = {range, type};
  thread_size_modifiers_.push_back(modifier);
}

// Adds a constraint to the list of constraints
void KernelInfo::AddConstraint(ConstraintFunction valid_if,
                               const std::vector<std::string> &parameters) {
  constraints_.push_back({valid_if, parameters});
}

// Sets the local memory size
void KernelInfo::SetLocalMemoryUsage(LocalMemoryFunction amount,
                                     const std::vector<std::string> &parameters) {
  local_memory_ = LocalMemory{amount, parameters};
}

// =================================================================================================

// Iterates over all modifiers (e.g. add a local multiplier) and applies these values to the
// global/local thread-sizes. Modified results are kept in temporary values, but are finally
// copied back to the member variables global_ and local_.
void KernelInfo::ComputeRanges(const Configuration &config) {

  // Initializes the result vectors
  size_t num_dimensions = global_base_.size();
  if (num_dimensions != local_base_.size()) {
    throw Exception("Mismatching number of global/local dimensions");
  }
  IntRange global_values(num_dimensions);
  IntRange local_values(num_dimensions);

  // Iterates over the three dimensions (x,y,z)
  for (size_t dim=0; dim<num_dimensions; ++dim) {
    global_values[dim] = global_base_[dim];
    local_values[dim] = local_base_[dim];

    // Iterates over all the applied modifiers
    for (auto &modifier: thread_size_modifiers_) {
      std::string modifier_string = modifier.value[dim];

      // Replaces the parameter-string with the corresponding integer and processes it
      bool found_string = false;
      for (auto &setting: config) {
        if (modifier_string == setting.name) {
          switch (modifier.type) {
            case ThreadSizeModifierType::kGlobalMul: global_values[dim] *= setting.value; break;
            case ThreadSizeModifierType::kGlobalDiv: global_values[dim] /= setting.value; break;
            case ThreadSizeModifierType::kLocalMul: local_values[dim] *= setting.value; break;
            case ThreadSizeModifierType::kLocalDiv: local_values[dim] /= setting.value; break;
            default: assert(0 && "Invalid modifier type");
          }
          found_string = true;
        }
      }

      // No replacement was found, there might be something wrong with the string
      if (!found_string && modifier_string != "") {
        throw Exception("Invalid modifier: "+modifier_string);
      }
    }
  }

  // Stores the final integer results
  global_ = global_values;
  local_ = local_values;
}

// Computes the number of iterations that kernel has to run based on the current configuration.
void KernelInfo::SetNumCurrentIterations(const Configuration &config) {
  bool found_string = false;
  std::string modifier_string = iterations_.parameter_name;

  for (auto &setting : config) {
    if (modifier_string == setting.name) {
      num_current_iterations_ = setting.value;
      found_string = true;
    }
  }

  // No replacement was found, there might be something wrong with the string
  if (!found_string && modifier_string != "") {
    throw Exception("Invalid kernel iterations modifier: " + modifier_string);
  }
}

// =================================================================================================

// Initializes an empty configuration (vector of name/value pairs) and kicks-off the recursive
// function to find all configurations. It also applies the user-defined constraints within.
void KernelInfo::SetConfigurations() {
  auto config = Configuration(parameters_.size());
  PopulateConfigurations(0, config);

  // Set default searcher to FullSearch.
  if (searcher_ == nullptr) {
    searcher_.reset(new FullSearch{ configurations() });
  }
}

// Iterates recursively over all permutations of the user-defined parameters. This code creates
// multiple chains, in which each chain selects a unique combination of values for all parameters.
// At the end of each chain (when all parameters are considered), the function stores the result
// into the configuration list.
void KernelInfo::PopulateConfigurations(const size_t index, const Configuration &config) {

  // End of the chain: all parameters are considered, store the resulting configuration if it is a
  // valid one according to the constraints
  if (index == parameters_.size()) {
    if (ValidConfiguration(config)) {
      configurations_.push_back(config);
    }
    return;
  }

  // This loop iterates over all values of the current parameter and calls this function
  // recursively
  Parameter parameter = parameters_[index];
  for (auto &value: parameter.values) {
    auto config_copy = config;
    config_copy[index] = Setting{parameter.name, value};
    PopulateConfigurations(index+1, config_copy);
  }
}

// Loops over all user-defined constraints to check whether or not the configuration is valid.
// Assumes initially all configurations are valid, then returns false if one of the constraints has
// not been met. Constraints consist of a user-defined function and a list of parameter names, which
// are replaced by parameter values in this function.
inline bool KernelInfo::ValidConfiguration(const Configuration &config) {

  // Iterates over all constraints
  for (auto &constraint: constraints_) {

    // Finds the values of the parameters
    auto values = std::vector<size_t>(constraint.parameters.size());
    for (auto i=size_t{0}; i<constraint.parameters.size(); ++i) {
      for (auto &setting: config) {
        if (setting.name == constraint.parameters[i]) {
          values[i] = setting.value;
          break;
        }
      }
    }

    // Checks this constraint for these values
    if (!constraint.valid_if(values)) {
      return false;
    }
  }

  // Computes the global and local worksizes
  ComputeRanges(config);

  // Verifies the global/local thread-sizes against device properties
  if (!device_.IsThreadConfigValid(local_)) { return false; };

  // Verifies the local memory usage
  std::vector<size_t> values_local_memory(0);
  for (auto &parameter: local_memory_.parameters) {
    for (auto &setting: config) {
      if (setting.name == parameter) {
        values_local_memory.push_back(setting.value);
        break;
      }
    }
  }
  if (local_memory_.parameters.size() != values_local_memory.size()) {
    throw Exception("Invalid settings for the local memory usage constraint");
  }
  auto local_mem_usage = local_memory_.amount(values_local_memory);
  if (!device_.IsLocalMemoryValid(local_mem_usage)) { return false; };

  // Everything was OK: this configuration is valid
  return true;
}

// =================================================================================================

// Methods that set searcher of the kernel.
void KernelInfo::UseFullSearch() {
  searcher_.reset(new FullSearch{ configurations() });
}

void KernelInfo::UseRandomSearch(const double fraction) {
  searcher_.reset(new RandomSearch{ configurations(), fraction });
}

void KernelInfo::UseAnnealing(const double fraction, const double max_temperature) {
  searcher_.reset(new Annealing{ configurations(), fraction, max_temperature });
}

void KernelInfo::UsePSO(const double fraction, const size_t swarm_size, const double influence_global,
                        const double influence_local, const double influence_random) {
  searcher_.reset(new PSO{ configurations(), parameters(), fraction, swarm_size, influence_global,
                           influence_local, influence_random });
}

// =================================================================================================

// Methods that add a new argument to the kernel.
void KernelInfo::AddArgumentInput(const MemArgument &argument) {
  arguments_input_.push_back(argument);
  argument_counter_++;
}

void KernelInfo::AddArgumentOutput(const MemArgument &argument) {
  arguments_output_.push_back(argument);
  argument_counter_++;
}

void KernelInfo::AddArgumentScalar(const short argument) {
  arguments_int_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const int argument) {
  arguments_int_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const size_t argument) {
  arguments_size_t_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const half argument) {
  arguments_float_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const float argument) {
  arguments_float_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const double argument) {
  arguments_double_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const float2 argument) {
  arguments_float2_.push_back({ argument_counter_++, argument });
}

void KernelInfo::AddArgumentScalar(const double2 argument) {
  arguments_double2_.push_back({ argument_counter_++, argument });
}

// =================================================================================================
} // namespace cltune
