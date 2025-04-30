#include "profile.h"

static bool _profile_enabled = false;
static std::map<std::string, double> _profile_times;
static std::map<std::string, int> _profile_counts;

void set_profile_enabled(bool enabled) {
  _profile_enabled = enabled;
}

bool get_profile_enabled() {
  return _profile_enabled;
}

const std::map<std::string, double>& profile_times() {
  return _profile_times;
}

const std::map<std::string, int>& profile_counts() {
  return _profile_counts;
}

OmpProfileGuard::OmpProfileGuard(const char* name) : _name(name) {
  _start = omp_get_wtime();
}

OmpProfileGuard::~OmpProfileGuard() {
  double end = omp_get_wtime();
  double duration = end - _start;
  if (_profile_enabled) {
    _profile_times[_name] += duration;
    _profile_counts[_name]++;
  }
}

ProfileDisabledGuard::ProfileDisabledGuard() {
  _was_enabled = get_profile_enabled();
  set_profile_enabled(false);
}

ProfileDisabledGuard::~ProfileDisabledGuard() {
  set_profile_enabled(_was_enabled);
}