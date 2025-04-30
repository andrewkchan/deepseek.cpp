#include <omp.h>
#include <map>
#include <string>

void set_profile_enabled(bool enabled);
bool get_profile_enabled();
const std::map<std::string, double>& profile_times();
const std::map<std::string, int>& profile_counts();

// This macro can be used to profile a block of code.
// Example:
// ```
// {
//   PROFILE_BLOCK(my_block);
//   // code to profile...
// }
// ```
// The execution time will be saved with key `my_block` in the profile_times map.
#define PROFILE_BLOCK(name) \
  OmpProfileGuard profile_guard(#name)

// This macro can be used to profile a single statement.
// Example:
// ```
// PROFILE(my_statement);
// ```
// The execution time will be saved with key `my_statement` in the profile_times map.
#define PROFILE(X) do { \
  PROFILE_BLOCK(X); \
  X; \
} while(0)

struct OmpProfileGuard {
public:
  OmpProfileGuard(const char* name);
  ~OmpProfileGuard();
private:
  const char* _name;
  double _start;
};

struct ProfileDisabledGuard {
  ProfileDisabledGuard();
  ~ProfileDisabledGuard();
private:
  bool _was_enabled;
};