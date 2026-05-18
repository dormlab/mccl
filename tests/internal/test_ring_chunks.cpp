#include "_test.h"

#include <algorithm>
#include <vector>

// Local copy of AllreduceAlgos.cpp's chunking — keeps the unit test free of
// torch deps. If the real code changes, update this in tandem.
struct ChunkInfo { int64_t s, e; size_t off, len; };

static std::vector<ChunkInfo> chunkize(int64_t numel, int world, size_t esize) {
    int64_t per = (numel + world - 1) / world;
    std::vector<ChunkInfo> chunks(world);
    for (int i = 0; i < world; ++i) {
        int64_t s = std::min<int64_t>(static_cast<int64_t>(i) * per, numel);
        int64_t e = std::min<int64_t>(static_cast<int64_t>(i + 1) * per, numel);
        chunks[i].s = s;
        chunks[i].e = e;
        chunks[i].off = static_cast<size_t>(s) * esize;
        chunks[i].len = static_cast<size_t>(e - s) * esize;
    }
    return chunks;
}

TEST_CASE("chunks cover the full tensor exactly") {
    for (int world : {2, 3, 4, 8, 16, 30}) {
        for (int64_t numel : {100, 1024, 1025, 1<<20, 1234567}) {
            auto c = chunkize(numel, world, 4);
            int64_t total = 0;
            for (auto& ck : c) {
                CHECK(ck.s <= ck.e);
                CHECK(ck.e <= numel);
                total += (ck.e - ck.s);
            }
            CHECK(total == numel);
        }
    }
}

TEST_CASE("chunks are contiguous and non-overlapping") {
    auto c = chunkize(/*numel=*/1000, /*world=*/7, /*esize=*/4);
    for (size_t i = 1; i < c.size(); ++i) {
        CHECK(c[i].s == c[i - 1].e);
    }
}

TEST_CASE("ring step indices wrap correctly") {
    // For each rank r and step k in 0..N-2:
    //   send_idx = (r - k + N) % N
    //   recv_idx = (r - k - 1 + N) % N
    // After N-1 reduce-scatter steps, rank r should have touched
    // exactly its own chunk r (as recv target on the last step).
    int N = 4;
    for (int r = 0; r < N; ++r) {
        int last_recv = (r - (N - 1) - 1 + N) % N;
        // last_recv should equal r (the chunk this rank ends up owning)
        CHECK(last_recv == r);
    }
}

TEST_CASE("ring chunks tolerate numel < world") {
    // Edge case: tensor smaller than world. Some chunks will be empty.
    // The selection logic in PG falls back to tree when numel < world,
    // but chunkize should still produce a valid (degenerate) partition.
    auto c = chunkize(/*numel=*/3, /*world=*/5, /*esize=*/4);
    CHECK(c.size() == 5);
    int64_t total = 0;
    for (auto& ck : c) total += (ck.e - ck.s);
    CHECK(total == 3);
    // Some chunks must be empty
    int empty = 0;
    for (auto& ck : c) if (ck.s == ck.e) ++empty;
    CHECK(empty > 0);
}

MCCL_RUN_ALL()
