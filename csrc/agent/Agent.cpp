#include "agent/Protocol.hpp"
#include "dmem/DistributedMemoryManager.hpp"
#include "metal/MPSInterop.hpp"
#include "metal/MetalKernels.hpp"
#include "metal/EventSync.hpp"
#include "common/Logging.hpp"
#include "common/Errors.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>
#include <csignal>

namespace distro {

static std::atomic<bool> g_running{true};
static std::unique_ptr<DistributedMemoryManager> g_dmem;

// ── TCP helpers ──────────────────────────────────────────────────────────

static int create_server_socket(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    DISTRO_CHECK(fd >= 0, "Agent: socket() failed");

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    DISTRO_CHECK(bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0,
                 "Agent: bind() failed");
    DISTRO_CHECK(listen(fd, 5) == 0, "Agent: listen() failed");

    DISTRO_INFO("Agent: listening on port %u", port);
    return fd;
}

static bool send_all(int fd, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

static bool recv_all(int fd, void* data, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(data);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

// ── Command handler ─────────────────────────────────────────────────────

static void handle_connection(int client_fd) {
    while (g_running.load()) {
        AgentHeader hdr{};
        if (!recv_all(client_fd, &hdr, AgentHeader::WIRE_SIZE)) break;

        AgentHeader resp_hdr{hdr.cmd, AgentStatus::OK, 0};

        switch (hdr.cmd) {
            case AgentCmd::PING: {
                // Nothing to do — OK already set
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                break;
            }

            case AgentCmd::REGISTER: {
                RegisterRequest req{};
                if (!recv_all(client_fd, &req, RegisterRequest::WIRE_SIZE)) {
                    resp_hdr.status = AgentStatus::ERR_MEMORY;
                    send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                    break;
                }

                void* ptr = reinterpret_cast<void*>(req.buffer_addr);
                if (!ptr || req.length == 0) {
                    resp_hdr.status = AgentStatus::ERR_MEMORY;
                    send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                    break;
                }

                try {
                    uint32_t region_id = g_dmem->register_region(ptr, req.length, req.flags);
                    RegisterResponse reg_resp{};
                    reg_resp.region_id = region_id;

                    // Look up the MR to get rkey
                    auto res = g_dmem->resolve(GlobalAddress{
                        g_dmem->node_id(), region_id, 0});
                    reg_resp.rkey = 0;  // rkey exposed via catalog, not here

                    resp_hdr.payload_len = RegisterResponse::WIRE_SIZE;
                    send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                    send_all(client_fd, &reg_resp, RegisterResponse::WIRE_SIZE);
                } catch (const std::exception& e) {
                    resp_hdr.status = AgentStatus::ERR_MEMORY;
                    send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                }
                break;
            }

            case AgentCmd::DEREGISTER: {
                uint32_t region_id = 0;
                if (!recv_all(client_fd, &region_id, sizeof(region_id))) break;
                g_dmem->unregister_region(region_id);
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                break;
            }

            case AgentCmd::SHADER_RUN: {
                ShaderRunRequest req{};
                if (!recv_all(client_fd, &req, ShaderRunRequest::WIRE_SIZE)) {
                    resp_hdr.status = AgentStatus::ERR_METAL;
                    send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                    break;
                }
                // Shader execution delegated to MetalKernels via head dispatch.
                // For now, agent just acks. Actual shader work is driven by
                // the head via distributed dispatch (Phase 3).
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                break;
            }

            case AgentCmd::SHADER_SYNC: {
                try {
                    metal_sync();
                } catch (...) {
                    resp_hdr.status = AgentStatus::ERR_METAL;
                }
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                break;
            }

            case AgentCmd::STATS: {
                StatsResponse s{};
                auto dm_stats = g_dmem->stats();
                s.total_put_bytes = dm_stats.total_put_bytes;
                s.total_get_bytes = dm_stats.total_get_bytes;
                s.total_put_ops   = dm_stats.total_put_ops;
                s.total_get_ops   = dm_stats.total_get_ops;
                s.region_count    = 0; // TODO: track in DMEM
                s.free_memory     = 0; // TODO: query MTLDevice
                s.gpu_temp_celsius = 0;
                s.link_state      = 1; // Assume up

                resp_hdr.payload_len = StatsResponse::WIRE_SIZE;
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                send_all(client_fd, &s, StatsResponse::WIRE_SIZE);
                break;
            }

            case AgentCmd::SHUTDOWN: {
                g_running = false;
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                break;
            }

            default: {
                resp_hdr.status = AgentStatus::ERR_INTERNAL;
                send_all(client_fd, &resp_hdr, AgentHeader::WIRE_SIZE);
                break;
            }
        }
    }
    close(client_fd);
}

// ── Entry point ─────────────────────────────────────────────────────────

int agent_main(int argc, char* argv[]) {
    uint16_t port = 9800;
    int num_peers = 3;

    if (argc > 1) port = static_cast<uint16_t>(std::atoi(argv[1]));
    if (argc > 2) num_peers = std::atoi(argv[2]);

    signal(SIGPIPE, SIG_IGN);

    // Init Metal kernels
    try {
        metal_kernels_init();
    } catch (const std::exception& e) {
        DISTRO_INFO("Agent: Metal init skipped (%s) — running headless", e.what());
    }

    // Init DMEM
    DistributedMemoryManager::Config cfg;
    cfg.node_id   = 0;  // Set by head via config command
    cfg.num_peers = num_peers;

    try {
        g_dmem = std::make_unique<DistributedMemoryManager>(cfg);
        g_dmem->start();
        DISTRO_INFO("Agent: DMEM started (node=%u, peers=%d)", cfg.node_id, num_peers);
    } catch (const std::exception& e) {
        DISTRO_INFO("Agent: DMEM init skipped (%s) — RDMA not available", e.what());
    }

    // TCP accept loop
    int server_fd = create_server_socket(port);

    while (g_running.load()) {
        struct sockaddr_in client_addr{};
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);

        if (client_fd < 0) {
            if (!g_running.load()) break;
            continue;
        }

        DISTRO_DEBUG("Agent: connection from %s:%d",
                     inet_ntoa(client_addr.sin_addr),
                     ntohs(client_addr.sin_port));

        std::thread(handle_connection, client_fd).detach();
    }

    close(server_fd);
    if (g_dmem) g_dmem->shutdown();
    DISTRO_INFO("Agent: shutdown complete");
    return 0;
}

} // namespace distro

#ifndef DISTRO_HEAD_BUILD
int main(int argc, char* argv[]) {
    return distro::agent_main(argc, argv);
}
#endif
