#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace distro {

class PeerMesh {
public:
    PeerMesh(c10::intrusive_ptr<c10d::Store> store,
             int rank, int world,
             std::chrono::milliseconds timeout);
    ~PeerMesh();

    void send(int peer, const void* buf, size_t len);
    void recv(int peer, void* buf, size_t len);

    int world() const { return world_; }
    int rank() const { return rank_; }

private:
    int rank_, world_;
    std::chrono::milliseconds timeout_;
    int listen_fd_ = -1;
    std::vector<int> peer_fds_;
    std::vector<std::unique_ptr<std::mutex>> send_mu_, recv_mu_;
};

} // namespace distro
