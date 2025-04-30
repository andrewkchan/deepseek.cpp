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

OmpProfileGuard::OmpProfileGuard(const char* name) : _name(name) {
  _start = omp_get_wtime();
}

OmpProfileGuard::~OmpProfileGuard() {
  double end = omp_get_wtime();
  double duration = end - _start;
  if (_profile_enabled) {
    std::string key = "";
    for (const auto& scope : _profile_scopes) {
      key += scope + ".";
    }
    key += _name;
    _profile_times[key] += duration;
  }
}

ProfileDisabledGuard::ProfileDisabledGuard() {
  _was_enabled = get_profile_enabled();
  set_profile_enabled(false);
}

ProfileDisabledGuard::~ProfileDisabledGuard() {
  set_profile_enabled(_was_enabled);
}

ProfileScope::ProfileScope(std::string name) : _name(name) {
  _profile_scopes.push_back(name);
}

ProfileScope::~ProfileScope() {
  _profile_scopes.pop_back();
}