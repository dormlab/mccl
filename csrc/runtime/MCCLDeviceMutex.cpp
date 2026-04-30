#include "runtime/MCCLDeviceMutex.hpp"

namespace distro {

std::recursive_mutex& distro_device_ops_mutex() {
    static std::recursive_mutex m;
    return m;
}

} // namespace distro
