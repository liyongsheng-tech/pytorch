#pragma once

// yf225 TODO: change all .at() to [] here!
// yf225 TODO: write test for this macro, especially the `ret.push_back` steps

#define FORWARD_HAS_DEFAULT_ARGS(...) \
 protected: \
  template <typename ModuleType, typename... ArgumentTypes> \
  friend class AnyModuleHolder; \
  bool _forward_has_default_args() override { \
    return true; \
  } \
  unsigned int _forward_num_required_args() override { \
    return __VA_ARGS__.front().key(); \
  } \
  std::vector<AnyValue> _forward_populate_default_args(std::vector<AnyValue>&& arguments) override { \
    unsigned int num_all_args = __VA_ARGS__[__VA_ARGS__.size() - 1].key() + 1; \
    TORCH_INTERNAL_ASSERT(arguments.size() >= _forward_num_required_args() && arguments.size() < num_all_args); \
    std::vector<AnyValue> ret; \
    ret.reserve(num_all_args); \
    for (size_t i = 0; i < arguments.size(); i++) { \
      ret.push_back(std::move(arguments.at(i))); \
    } \
    for (auto& arg_info : __VA_ARGS__.items()) \
      if (arg_info.key() > ret.size() - 1) ret.push_back(std::move(arg_info.value())); \
    return std::move(ret); \
  }
