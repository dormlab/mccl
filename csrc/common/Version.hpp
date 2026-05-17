#pragma once

#define DISTRO_VERSION_MAJOR 0
#define DISTRO_VERSION_MINOR 4
#define DISTRO_VERSION_PATCH 0

#define DISTRO_PROTOCOL_VERSION 1

#define DISTRO_VERSION_STRING "0.4.0"

namespace mccl {

struct Version {
    static constexpr int major   = DISTRO_VERSION_MAJOR;
    static constexpr int minor   = DISTRO_VERSION_MINOR;
    static constexpr int patch   = DISTRO_VERSION_PATCH;
    static constexpr int protocol = DISTRO_PROTOCOL_VERSION;
};

} // namespace mccl
