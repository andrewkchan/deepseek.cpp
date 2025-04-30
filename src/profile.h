#include <omp.h>
#include <map>
#include <string>

void set_profile_enabled(bool enabled);
bool get_profile_enabled();
const std::map<std::string, double>& profile_times();

// This macro can be used to profile a block of code.
// Example:
// ```
// {
//   PROFILE_BLOCK(my_block);
//   // code to profile...
// }
// ```
// The execution time will be saved with key `my_block` in the profile_times map.
// `my_block` need not be a variable name; it can be any string.
#define PROFILE_BLOCK(name) \
  ProfileScope profile_scope(#name)

// This macro can be used to profile a single statement.
// Example:
// ```
// PROFILE(my_statement);
// ```
// The execution time will be saved with key `my_statement` in the profile_times map.
// `my_statement` should be a valid C++ statement or expression.
#define PROFILE(X) do { \
  PROFILE_BLOCK(X); \
  X; \
} while(0)

struct ProfileScope {
  ProfileScope(std::string name);
  ProfileScope(const char* name);
  ~ProfileScope();
private:
  double _start;
};

struct ProfileDisabledScope {
  ProfileDisabledScope();
  ~ProfileDisabledScope();
private:
  bool _was_enabled;
};