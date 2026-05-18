#include "backend/PeerMesh.hpp"
#include "common/Errors.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace mccl {

namespace {



const std::vector<std::string>& prefix_priority() {
    static const std::vector<std::string> v = [] {
        const char* env = std::getenv("MCCL_IFACE_PRIORITY");
        std::vector<std::string> out;
        if (!env || !*env) return out;
        std::string s = env;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) out.emplace_back(s.substr(i, j - i));
            i = j + 1;
        }
        return out;
    }();
    return v;
}

int score(const std::string& ip) {
    const auto& pri = prefix_priority();
    for (size_t i = 0; i < pri.size(); i++) {
        if (ip.rfind(pri[i], 0) == 0) return static_cast<int>(i);
    }
    return static_cast<int>(pri.size());
}

std::string subnet24(const std::string& ip) {
    int dots = 0;
    for (size_t i = 0; i < ip.size(); i++) {
        if (ip[i] == '.') {
            if (++dots == 3) return ip.substr(0, i);
        }
    }
    return ip;
}

std::vector<std::string> local_ips() {
    std::vector<std::string> v;
    struct ifaddrs* ifa = nullptr;
    if (getifaddrs(&ifa) != 0) return v;
    for (auto* p = ifa; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        if (!(p->ifa_flags & IFF_UP) || (p->ifa_flags & IFF_LOOPBACK)) continue;
        char buf[INET_ADDRSTRLEN];
        auto* sin = reinterpret_cast<sockaddr_in*>(p->ifa_addr);
        if (!inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) continue;
        v.emplace_back(buf);
    }
    freeifaddrs(ifa);
    std::sort(v.begin(), v.end(),
              [](const std::string& a, const std::string& b) {
                  return score(a) < score(b);
              });
    return v;
}

void tune(int fd) {
    int big = 16 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

bool write_all(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    while (len) {
        ssize_t n = ::send(fd, p, len, 0);
        if (n <= 0) return false;
        p += n; len -= n;
    }
    return true;
}

bool read_all(int fd, void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    while (len) {
        ssize_t n = ::recv(fd, p, len, 0);
        if (n <= 0) return false;
        p += n; len -= n;
    }
    return true;
}

} // namespace

PeerMesh::PeerMesh(c10::intrusive_ptr<c10d::Store> store,
                   int rank, int world,
                   std::chrono::milliseconds timeout)
    : rank_(rank), world_(world), timeout_(timeout) {
    peer_fds_.assign(world_, -1);
    send_mu_.resize(world_);
    recv_mu_.resize(world_);
    for (int i = 0; i < world_; i++) {
        send_mu_[i] = std::make_unique<std::mutex>();
        recv_mu_[i] = std::make_unique<std::mutex>();
    }

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    DISTRO_CHECK(listen_fd_ >= 0, "PeerMesh socket");
    int yes = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    DISTRO_CHECK(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
                 "PeerMesh bind");
    DISTRO_CHECK(::listen(listen_fd_, world_) == 0, "PeerMesh listen");
    socklen_t alen = sizeof(addr);
    DISTRO_CHECK(getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &alen) == 0,
                 "PeerMesh getsockname");
    int port = ntohs(addr.sin_port);

    auto ips_all = local_ips();
    // Only advertise IPs whose /24 prefix is in MCCL_IFACE_PRIORITY. This
    // keeps Wi-Fi / Ethernet interfaces out of the peer-discovery exchange
    // entirely, so a peer can never pick them.
    std::vector<std::string> ips;
    {
        const auto& pri = prefix_priority();
        for (const auto& ip : ips_all) {
            if (pri.empty()) { ips.push_back(ip); continue; }
            for (const auto& p : pri) {
                if (ip.rfind(p, 0) == 0) { ips.push_back(ip); break; }
            }
        }
    }
    DISTRO_CHECK(!ips.empty(),
                 "PeerMesh: no local IP matches MCCL_IFACE_PRIORITY; check "
                 "that Thunderbolt interfaces are up and the priority list "
                 "covers their subnets");
    std::vector<std::string> my_subnets;
    my_subnets.reserve(ips.size());
    for (const auto& ip : ips) my_subnets.emplace_back(subnet24(ip));
    std::ostringstream os;
    for (size_t i = 0; i < ips.size(); i++) {
        if (i) os << ",";
        os << ips[i];
    }
    os << ":" << port;
    std::string ep = os.str();
    std::vector<uint8_t> ev(ep.begin(), ep.end());
    store->set("mccl/peer/" + std::to_string(rank_), ev);

    std::vector<std::string> eps(world_);
    eps[rank_] = ep;
    for (int p = 0; p < world_; p++) {
        if (p == rank_) continue;
        auto k = "mccl/peer/" + std::to_string(p);
        store->wait({k}, timeout_);
        auto b = store->get(k);
        eps[p] = std::string(b.begin(), b.end());
    }

    int pending_accepts = 0;
    for (int p = 0; p < rank_; p++) pending_accepts++;

    std::thread acceptor([&]() {
        for (int i = 0; i < pending_accepts; i++) {
            sockaddr_in cli{}; socklen_t cl = sizeof(cli);
            int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&cli), &cl);
            if (fd < 0) continue;
            int peer = -1;
            if (!read_all(fd, &peer, 4) || peer < 0 || peer >= rank_) {
                ::close(fd); continue;
            }
            int self = rank_;
            if (!write_all(fd, &self, 4)) { ::close(fd); continue; }
            tune(fd);
            peer_fds_[peer] = fd;
        }
    });

    for (int p = rank_ + 1; p < world_; p++) {
        const auto& s = eps[p];
        auto colon = s.rfind(':');
        DISTRO_CHECK(colon != std::string::npos, "PeerMesh: bad endpoint");
        int peer_port = std::atoi(s.c_str() + colon + 1);
        std::vector<std::string> cands;
        size_t i = 0;
        while (i < colon) {
            auto j = s.find(',', i);
            if (j == std::string::npos || j > colon) j = colon;
            cands.emplace_back(s.substr(i, j - i));
            i = j + 1;
        }
        std::vector<std::string> reachable;
        reachable.reserve(cands.size());
        const auto& pri = prefix_priority();
        for (const auto& ip : cands) {
            auto sub = subnet24(ip);
            bool same_subnet = false;
            for (const auto& mine : my_subnets) {
                if (sub == mine) { same_subnet = true; break; }
            }
            if (!same_subnet) continue;
            // Hard-filter against the priority list. Without this, PeerMesh
            // would silently fall back to non-TB interfaces (Wi-Fi /
            // Ethernet) when a TB pair fails, dragging the whole ring down
            // to ~1 Gbps. We want loud failures, not silent slow paths.
            if (!pri.empty()) {
                bool in_pri = false;
                for (const auto& p : pri) {
                    if (ip.rfind(p, 0) == 0) { in_pri = true; break; }
                }
                if (!in_pri) continue;
            }
            reachable.push_back(ip);
        }
        DISTRO_CHECK(!reachable.empty(),
                     "PeerMesh: no MCCL_IFACE_PRIORITY-matching subnet shared "
                     "with peer " + std::to_string(p) +
                     "; refusing to fall back to a non-priority interface");
        std::sort(reachable.begin(), reachable.end(),
                  [](const std::string& a, const std::string& b) {
                      return score(a) < score(b);
                  });

        int fd = -1;
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        while (fd < 0 && std::chrono::steady_clock::now() < deadline) {
            for (auto& ip : reachable) {
                int s_fd = ::socket(AF_INET, SOCK_STREAM, 0);
                if (s_fd < 0) continue;
                sockaddr_in a{};
                a.sin_family = AF_INET;
                a.sin_port = htons(peer_port);
                if (inet_pton(AF_INET, ip.c_str(), &a.sin_addr) != 1) {
                    ::close(s_fd); continue;
                }
                timeval tv{1, 0};
                setsockopt(s_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                setsockopt(s_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
                if (::connect(s_fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
                    ::close(s_fd); continue;
                }
                int self = rank_;
                if (!write_all(s_fd, &self, 4)) { ::close(s_fd); continue; }
                int got = -1;
                if (!read_all(s_fd, &got, 4) || got != p) { ::close(s_fd); continue; }
                timeval zero{0, 0};
                setsockopt(s_fd, SOL_SOCKET, SO_RCVTIMEO, &zero, sizeof(zero));
                setsockopt(s_fd, SOL_SOCKET, SO_SNDTIMEO, &zero, sizeof(zero));
                tune(s_fd);
                fd = s_fd;
                break;
            }
            if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        DISTRO_CHECK(fd >= 0, "PeerMesh: failed to connect to peer "
                              + std::to_string(p));
        peer_fds_[p] = fd;
    }

    acceptor.join();
    for (int p = 0; p < world_; p++) {
        if (p == rank_) continue;
        DISTRO_CHECK(peer_fds_[p] >= 0,
                     "PeerMesh: missing socket for peer " + std::to_string(p));
    }
}

PeerMesh::~PeerMesh() {
    for (int fd : peer_fds_) if (fd >= 0) ::close(fd);
    if (listen_fd_ >= 0) ::close(listen_fd_);
}

void PeerMesh::send(int peer, const void* buf, size_t len) {
    std::lock_guard<std::mutex> lk(*send_mu_[peer]);
    DISTRO_CHECK(write_all(peer_fds_[peer], buf, len), "PeerMesh: send failed");
}

void PeerMesh::recv(int peer, void* buf, size_t len) {
    std::lock_guard<std::mutex> lk(*recv_mu_[peer]);
    DISTRO_CHECK(read_all(peer_fds_[peer], buf, len), "PeerMesh: recv failed");
}

} // namespace mccl
