// tests for --numa split: the topology parser, node-local allocation, and NUMA backends
//
// the parser is pure, so it runs against generated sysfs trees and needs no NUMA hardware

#include "ggml-cpu-numa.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace ggml::cpu::numa;

namespace fs = std::filesystem;

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-58s %s\n", what.c_str(), ok ? "OK" : "FAILED");
    if (!ok) {
        n_fail++;
    }
}

static void write_file(const fs::path & path, const std::string & content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << content;
}

struct fake_node {
    int         id;
    std::string cpulist;
    size_t      mem_total_kb;
};

// build a sysfs tree, every CPU gets a sibling at cpu+n_cpus_per_core_offset to emulate SMT
static fs::path make_sysfs(const std::string & name, const std::string & online, const std::vector<fake_node> & nodes, int smt_offset) {
    const fs::path root = fs::temp_directory_path() / ("ggml-numa-test-" + name);
    fs::remove_all(root);

    write_file(root / "devices/system/node/online", online + "\n");

    for (const auto & n : nodes) {
        const fs::path dir = root / ("devices/system/node/node" + std::to_string(n.id));
        write_file(dir / "cpulist", n.cpulist + "\n");
        write_file(dir / "meminfo",
                "Node " + std::to_string(n.id) + " MemTotal:       " + std::to_string(n.mem_total_kb) + " kB\n"
                "Node " + std::to_string(n.id) + " MemFree:        " + std::to_string(n.mem_total_kb / 4) + " kB\n"
                "Node " + std::to_string(n.id) + " Active(file):   " + std::to_string(n.mem_total_kb / 8) + " kB\n"
                "Node " + std::to_string(n.id) + " Inactive(file): " + std::to_string(n.mem_total_kb / 4) + " kB\n");

        for (int cpu : parse_list(n.cpulist)) {
            const int sibling = cpu < smt_offset ? cpu + smt_offset : cpu - smt_offset;
            write_file(root / ("devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list"),
                    std::to_string(std::min(cpu, sibling)) + "," + std::to_string(std::max(cpu, sibling)) + "\n");
        }
    }

    return root;
}

int main() {
    printf("parse_list\n");
    {
        check(parse_list("0-3") == std::vector<int>({0, 1, 2, 3}), "range");
        check(parse_list("0,2,4") == std::vector<int>({0, 2, 4}), "comma separated");
        check(parse_list("0-1,4,6-7\n") == std::vector<int>({0, 1, 4, 6, 7}), "mixed with newline");
        check(parse_list("5") == std::vector<int>({5}), "single value");
        check(parse_list("").empty(), "empty");
        check(parse_list("garbage").empty(), "garbage");
    }

    printf("parse_topology\n");
    {
        // emulated topology: 2 nodes, interleaved CPU numbering, hyperthreads at +48
        const fs::path root = make_sysfs("2node", "0-1", {{0, "0-1,48-49", 8000}, {1, "2-3,50-51", 8000}}, 48);

        const auto nodes = parse_topology(root.string(), {});
        check(nodes.size() == 2, "two nodes found");
        if (nodes.size() == 2) {
            check(nodes[0].id == 0 && nodes[1].id == 1, "node ids");
            check(nodes[0].cpus == std::vector<int>({0, 1, 48, 49}), "node 0 cpus");
            check(nodes[1].cpus == std::vector<int>({2, 3, 50, 51}), "node 1 cpus");
            check(nodes[0].n_cores == 2 && nodes[1].n_cores == 2, "hyperthreads counted as one core");
            check(nodes[0].mem_total == 8000ull * 1024, "memory in bytes");
            // MemFree plus Inactive(file), the reclaimable page cache; Active(file) is not counted
            check(nodes[0].mem_available == (2000ull + 2000ull) * 1024, "available includes reclaimable cache");
        }
    }
    {
        const fs::path root  = make_sysfs("1node", "0", {{0, "0-3", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1, "single node");
    }
    {
        // node 1 offline, so the ids have a hole
        const fs::path root  = make_sysfs("hole", "0,2", {{0, "0-1", 8000}, {2, "4-5", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 2, "hole in node numbering");
        if (nodes.size() == 2) {
            check(nodes[1].id == 2, "second node keeps its id");
        }
    }
    {
        // a memory-only node, e.g. CXL, cannot run threads
        const fs::path root  = make_sysfs("nocpu", "0-1", {{0, "0-1", 8000}, {1, "", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1 && nodes[0].id == 0, "node without cpus dropped");
    }
    {
        const fs::path root  = make_sysfs("nomem", "0-1", {{0, "0-1", 8000}, {1, "2-3", 0}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1 && nodes[0].id == 0, "node without memory dropped");
    }
    {
        const fs::path root = make_sysfs("affinity", "0-1", {{0, "0-1", 8000}, {1, "2-3", 8000}}, 2);

        const auto pinned = parse_topology(root.string(), {2, 3});
        check(pinned.size() == 1 && pinned[0].id == 1, "affinity mask leaves one node");

        const auto partial = parse_topology(root.string(), {0, 2});
        check(partial.size() == 2 && partial[0].cpus == std::vector<int>({0}), "affinity mask trims cpus");
    }
    {
        const auto nodes = parse_topology("/nonexistent-sysfs", {});
        check(nodes.empty(), "missing sysfs is not an error");
    }
    {
        std::vector<fake_node> many;
        for (int i = 0; i < 16; i++) {
            many.push_back({i, std::to_string(2 * i) + "-" + std::to_string(2 * i + 1), 8000});
        }
        const fs::path root  = make_sysfs("16node", "0-15", many, 1);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 16, "sixteen nodes");
    }

    printf("node local allocation\n");
    if (topology().size() < 2) {
        printf("  skipped, this machine has fewer than 2 usable NUMA nodes\n");
    } else {
        const size_t size = 64ull << 20;

        for (const auto & n : topology()) {
            std::string error;

            void * data = alloc_onnode(size, n.id, error);
            check(data != nullptr, "allocated 64 MiB on node " + std::to_string(n.id) + (data ? "" : ": " + error));
            if (data == nullptr) {
                continue;
            }

            // the allocator only samples pages, so check the whole range here
            size_t n_remote = 0;
            for (size_t off = 0; off < size; off += 2u << 20) {
                ((char *) data)[off] = 1;
                if (page_node((char *) data + off) != n.id) {
                    n_remote++;
                }
            }
            check(n_remote == 0, "every page of node " + std::to_string(n.id) + " is local");

            free_onnode(data, size);
        }

        std::string error;
        check(alloc_onnode(size, 999, error) == nullptr && !error.empty(), "allocation on an unusable node fails");
    }

    printf("numa split registration & backend verification\n");
    {
        const enum llama_numa_init_status status = llama_numa_init_ex(GGML_NUMA_STRATEGY_SPLIT);

        if (topology().size() < 2) {
            check(status == LLAMA_NUMA_INIT_STATUS_UNAVAILABLE, "unavailable with fewer than 2 nodes");
        } else {
            check(status == LLAMA_NUMA_INIT_STATUS_SUCCESS, "split initialized");
            check(is_numa_split(), "is_numa_split reports true");

            for (const auto & n : topology()) {
                const std::string expected = "CPU" + std::to_string(n.id);

                ggml_backend_buffer_type_t buft = ggml_backend_cpu_numa_buffer_type(n.id);
                check(buft != nullptr, expected + " buffer type found");
                if (!buft) continue;

                check(std::string(ggml_backend_buft_name(buft)) == expected, expected + " has correct name");
                check(ggml_backend_buft_is_host(buft), expected + " buffer type is host memory");

                ggml_backend_t backend = ggml_backend_cpu_numa_init(n.id);
                check(backend != nullptr, expected + " backend initialized");
                if (!backend) continue;

                check(ggml_backend_is_cpu(backend), expected + " is CPU backend");
                check(ggml_backend_is_cpu_numa(backend), expected + " is CPU NUMA backend");
                check(ggml_backend_cpu_numa_get_node(backend) == n.id, expected + " reports node id");
                check(ggml_backend_supports_buft(backend, buft), expected + " supports its own buffer type");

                // Test buffer allocation on node
                const size_t buf_size = 2 * 1024 * 1024;
                ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, buf_size);
                check(buf != nullptr, expected + " allocated 2 MiB buffer");
                if (buf) {
                    void * base = ggml_backend_buffer_get_base(buf);
                    check(base != nullptr, expected + " buffer base ptr valid");
                    if (base) {
                        memset(base, 0x42, buf_size);
                        check(page_node(base) == n.id, expected + " buffer memory is verified local");
                    }
                    ggml_backend_buffer_free(buf);
                }

                // Verify cross-node rejection
                for (const auto & other : topology()) {
                    if (other.id == n.id) continue;
                    ggml_backend_buffer_type_t other_buft = ggml_backend_cpu_numa_buffer_type(other.id);
                    check(!ggml_backend_supports_buft(backend, other_buft),
                          expected + " rejects buffer type of CPU" + std::to_string(other.id));
                }

                ggml_backend_free(backend);
            }

            check(llama_numa_init_ex(GGML_NUMA_STRATEGY_SPLIT) == LLAMA_NUMA_INIT_STATUS_SUCCESS, "second call is idempotent");
        }
    }

    printf("topology of this machine\n");
    for (const auto & n : topology()) {
        printf("  node %d: %zu cpus, %d cores, %zu MiB total, %zu MiB free\n",
                n.id, n.cpus.size(), n.n_cores, n.mem_total >> 20, n.mem_available >> 20);
    }

    printf("%s\n", n_fail == 0 ? "test-numa: all tests OK" : "test-numa: FAILED");

    return n_fail == 0 ? 0 : 1;
}
