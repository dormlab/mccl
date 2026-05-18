// Minimal single-header test runner for mccl internal C++ tests.
// 30 lines, no deps. Drop a .cpp in this dir, write TEST_CASE blocks,
// and end with MCCL_RUN_ALL().
#pragma once
#include <cstdio>
#include <vector>
#include <string>

namespace mccl_test {
struct Case { const char* name; void (*fn)(); };
inline std::vector<Case>& cases() { static std::vector<Case> v; return v; }
inline int& fail_count() { static int n = 0; return n; }
}

#define MCCL_CAT_(a, b) a##b
#define MCCL_CAT(a, b)  MCCL_CAT_(a, b)

#define TEST_CASE(name)                                                       \
    static void MCCL_CAT(_t_, __LINE__)();                                    \
    static int  MCCL_CAT(_r_, __LINE__) =                                     \
        (::mccl_test::cases().push_back({name, MCCL_CAT(_t_, __LINE__)}), 0); \
    static void MCCL_CAT(_t_, __LINE__)()

#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "  FAIL %s:%d  %s\n",                        \
                         __FILE__, __LINE__, #cond);                          \
            ::mccl_test::fail_count()++;                                      \
        }                                                                     \
    } while (0)

#define MCCL_RUN_ALL()                                                        \
    int main() {                                                              \
        int before;                                                           \
        for (auto& c : ::mccl_test::cases()) {                                \
            before = ::mccl_test::fail_count();                               \
            std::printf("[RUN ] %s\n", c.name);                               \
            c.fn();                                                           \
            std::printf("%s %s\n",                                            \
                ::mccl_test::fail_count() == before ? "[ OK ]" : "[FAIL]",    \
                c.name);                                                      \
        }                                                                     \
        std::printf("%d failures\n", ::mccl_test::fail_count());              \
        return ::mccl_test::fail_count() ? 1 : 0;                             \
    }
