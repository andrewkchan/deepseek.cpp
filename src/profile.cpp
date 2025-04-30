#include "profile.h"

#include <vector>

static bool _profile_enabled = true;
static std::vector<std::string> _profile_scopes;
static std::map<std::string, double> _profile_times;

void set_profile_enabled(bool enabled) {
  _profile_enabled = enabled;
}

bool get_profile_enabled() {
  return _profile_enabled;
}

const std::map<std::string, double>& profile_times() {
  return _profile_times;
}

#if PROFILE_ENABLED
ProfileScope::ProfileScope(std::string name) {
  _profile_scopes.push_back(name);
  _start = omp_get_wtime();
}

ProfileScope::ProfileScope(const char* name) : 
  ProfileScope(std::string(name)) {}

ProfileScope::~ProfileScope() {
  double end = omp_get_wtime();
  double duration = end - _start;
  if (_profile_enabled) {
    std::string key = "";
    for (const auto& scope : _profile_scopes) {
      key += scope + ".";
    }
    _profile_times[key] += duration;
  }
  _profile_scopes.pop_back();
}
#else
ProfileScope::ProfileScope(std::string name) {}
ProfileScope::ProfileScope(const char* name) {}
ProfileScope::~ProfileScope() {}
#endif

ProfileDisabledScope::ProfileDisabledScope() {
  _was_enabled = get_profile_enabled();
  set_profile_enabled(false);
}

ProfileDisabledScope::~ProfileDisabledScope() {
  set_profile_enabled(_was_enabled);
}