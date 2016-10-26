
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the KernelInfo class which holds information for a single kernel including
// all its parameters and settings. It holds the kernel name and source-code as a string, it holds
// the global and local NDRange settings, and it holds the parameters set by the user (and its
// permutations).
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

#ifndef CLTUNE_KERNEL_INFO_H_
#define CLTUNE_KERNEL_INFO_H_

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <complex> // std::complex

// Uses either the OpenCL or CUDA back-end (CLCudaAPI C++11 headers)
#if USE_OPENCL
  #include "internal/clpp11.h"
#else
  #include "internal/cupp11.h"
#endif

#include "cltune.h"
#include "internal/half.h" // host data-type for half-precision floating-point (16-bit)

namespace cltune {
// =================================================================================================

// Shorthands for complex data-types
  using float2 = std::complex<float>; // cl_float2;
  using double2 = std::complex<double>; // cl_double2;

// Raw device buffer
#if USE_OPENCL
  using BufferRaw = cl_mem;
#else
  using BufferRaw = CUdeviceptr;
#endif

// Enumeration of currently supported data-types for device memory arguments
enum class MemType { kShort, kInt, kSizeT, kHalf, kFloat, kDouble, kFloat2, kDouble2 };

// See comment at top of file for a description of the class
class KernelInfo {
 public:

  // Enumeration of modifiers to global/local thread-sizes
  enum class ThreadSizeModifierType { kGlobalMul, kGlobalDiv, kLocalMul, kLocalDiv };

  // Helper structure holding a parameter name and a list of all values
  struct Parameter {
    std::string name;
    std::vector<size_t> values;
  };

  // Helper structure to store a device memory argument for a kernel
  struct MemArgument {
    size_t index;       // The kernel-argument index
    size_t size;        // The number of elements (not bytes)
    MemType type;       // The data-type (e.g. float)
    BufferRaw buffer;   // The buffer on the device
  };

  // Helper structure holding a setting: a name and a value. Multiple settings combined make a
  // single configuration.
  struct Setting {
    std::string name;
    size_t value;
    std::string GetDefine() const { return "#define "+name+" "+GetValueString()+"\n"; }
    std::string GetConfig() const { return name+" "+GetValueString(); }
    std::string GetDatabase() const { return "{\""+name+"\","+GetValueString()+"}"; }
    std::string GetValueString() const { return std::to_string(static_cast<long long>(value)); }
  };
  using Configuration = std::vector<Setting>;

  // Helper structure holding a modifier: its value and its type
  struct ThreadSizeModifier {
    StringRange value;
    ThreadSizeModifierType type;
  };

  // Helper structure for number of kernel iterations
  struct IterationsModifier {
    std::vector<size_t> valid_iterations;
    std::string parameter_name;
  };

  // Helper structure holding a constraint on parameters. This constraint consists of a constraint
  // function object and a vector of paramater names represented as strings.
  struct Constraint {
    ConstraintFunction valid_if;
    std::vector<std::string> parameters;
  };

  // As above, but for local memory size.
  struct LocalMemory {
    LocalMemoryFunction amount;
    std::vector<std::string> parameters;
  };

  // Exception of the KernelInfo class
  class Exception : public std::runtime_error {
   public:
    Exception(const std::string &message): std::runtime_error(message) { }
  };

  // Initializes the class with a given name and a string of kernel source-code
  explicit KernelInfo(const std::string name, const std::string source, const Device &device);
  ~KernelInfo();

  // Accessors (getters)
  std::string name() const { return name_; }
  std::string source() const { return source_; }
  std::vector<Parameter> parameters() const { return parameters_; }
  IterationsModifier iterations() const { return iterations_; }
  size_t num_current_iterations() const { return num_current_iterations_; }
  IntRange global_base() const { return global_base_; }
  IntRange local_base() const { return local_base_; }
  IntRange global() const { return global_; }
  IntRange local() const { return local_; }
  std::vector<Configuration> configurations() { return configurations_; }
  size_t argument_counter() const { return argument_counter_; }
  std::vector<MemArgument> arguments_input() const { return arguments_input_; }
  std::vector<MemArgument> arguments_output() const { return arguments_output_; }
  std::vector<std::pair<size_t, int>> arguments_int() const { return arguments_int_; }
  std::vector<std::pair<size_t, size_t>> arguments_size_t() const { return arguments_size_t_; }
  std::vector<std::pair<size_t, float>> arguments_float() const { return arguments_float_; }
  std::vector<std::pair<size_t, double>> arguments_double() const { return arguments_double_; }
  std::vector<std::pair<size_t, float2>> arguments_float2() const { return arguments_float2_; }
  std::vector<std::pair<size_t, double2>> arguments_double2() const { return arguments_double2_; }

  // Accessors (setters) - Note that these also pre-set the final global/local size
  void set_global_base(IntRange global) { global_base_ = global; global_ = global; }
  void set_local_base(IntRange local) { local_base_ = local; local_ = local; }
  void set_iterations(std::vector<size_t> valid_iterations, std::string parameter_name) {
    iterations_.valid_iterations = valid_iterations;
    iterations_.parameter_name = parameter_name;
  }

  // Prepend to the source-code
  void PrependSource(const std::string &extra_source);

  // Adds a new parameter with a name and a vector of possible values
  void AddParameter(const std::string &name, const std::vector<size_t> &values);

  // Checks wheter a parameter exists, returns "true" if it does exist
  bool ParameterExists(const std::string parameter_name);

  // Specifies a modifier in the form of a StringRange to the global/local thread-sizes. This
  // modifier has to contain (per-dimension) the name of a single parameter or an empty string. The
  // supported modifiers are given by the ThreadSizeModifierType enumeration.
  void AddModifier(const StringRange range, const ThreadSizeModifierType type);

  // Adds a new constraint to the set of parameters (e.g. must be equal or larger than). The
  // constraints come in the form of a function object which takes a number of tuning parameters,
  // given as a vector of strings (parameter names). Their names are later substituted by actual
  // values.
  void AddConstraint(ConstraintFunction valid_if, const std::vector<std::string> &parameters);

  // As above, but for local memory usage
  void SetLocalMemoryUsage(LocalMemoryFunction amount, const std::vector<std::string> &parameters);

  // Computes the global/local ranges (in NDRange-form) based on all global/local thread-sizes (in
  // StringRange-form) and a single permutation (i.e. a configuration) containing a list of all
  // parameter names and their current values.
  void ComputeRanges(const Configuration &config);

  // Computes the number of iterations that kernel has to run based on the current configuration.
  void SetNumCurrentIterations(const Configuration &config);

  // Computes all permutations based on the parameters and their values (the configuration list).
  // The result is stored as a member variable.
  void SetConfigurations();

  // Methods that add a new argument to a kernel.
  void AddArgumentInput(const MemArgument &argument);
  void AddArgumentOutput(const MemArgument &argument);
  void AddArgumentScalar(const short argument);
  void AddArgumentScalar(const int argument);
  void AddArgumentScalar(const size_t argument);
  void AddArgumentScalar(const half argument);
  void AddArgumentScalar(const float argument);
  void AddArgumentScalar(const double argument);
  void AddArgumentScalar(const float2 argument);
  void AddArgumentScalar(const double2 argument);
  
 private:
  // Called recursively internally by SetConfigurations 
  void PopulateConfigurations(const size_t index, const Configuration &config);

  // Returns whether or not a given configuration is valid. This check is based on the user-supplied
  // constraints.
  bool ValidConfiguration(const Configuration &config);

  // Member variables
  std::string name_;
  std::string source_;
  std::vector<Parameter> parameters_;
  std::vector<Configuration> configurations_;
  std::vector<Constraint> constraints_;
  LocalMemory local_memory_;
  IterationsModifier iterations_;
  size_t num_current_iterations_;

  Device device_;

  // Global/local thread-sizes
  IntRange global_base_;
  IntRange local_base_;
  IntRange global_;
  IntRange local_;

  // Storage of kernel arguments
  size_t argument_counter_;
  std::vector<MemArgument> arguments_input_;
  std::vector<MemArgument> arguments_output_;
  std::vector<std::pair<size_t, int>> arguments_int_;
  std::vector<std::pair<size_t, size_t>> arguments_size_t_;
  std::vector<std::pair<size_t, float>> arguments_float_;
  std::vector<std::pair<size_t, double>> arguments_double_;
  std::vector<std::pair<size_t, float2>> arguments_float2_;
  std::vector<std::pair<size_t, double2>> arguments_double2_;

  // Multipliers and dividers for global/local thread-sizes
  std::vector<ThreadSizeModifier> thread_size_modifiers_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_KERNEL_INFO_H_
#endif
