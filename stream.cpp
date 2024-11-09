#include <iostream>
#include "common/core_coord.h"
#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "common/bfloat16.hpp"
#include "common/work_split.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <string_view>
#include <vector>
#include <chrono>

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_SIZE = TILE_WIDTH * TILE_HEIGHT;
constexpr uint32_t TILE_SIZE_BYTES = TILE_SIZE * sizeof(bfloat16);

namespace stream {
    class Kernel {
      public:
        Kernel() = default;

        // Add a new input or output port to the kernel
        void add_input_port(const std::string& name, tt::CB cb) {
            input_ports[name] = cb;
        }

        void add_output_port(const std::string& name, tt::CB cb) {
            output_ports[name] = cb;
        }

        void generate_kernels() {
            auto reader_code = generate_reader_kernel();
            const std::string generated_reader_kernel_path = "tt_metal/programming_examples/personal/stream/kernels/generated/reader.cpp";
            auto reader_kernel_file = std::ofstream(generated_reader_kernel_path);
            if (!reader_kernel_file.is_open()) {
                tt::log_error("Failed to open file for writing: {}", generated_reader_kernel_path);
            }
            reader_kernel_file << reader_code;
            reader_kernel_file.close();
        }
      private:
        // TODO: For each input port, we need need a CB to move data from NOC to compute.
        // If the kernel is a generator, each input port will be reading from a DRAM buffer.
        // If our input/output ports are connected to other kernels, need to determine how to do
        // the pipelining.
        std::unordered_map<std::string, tt::CB> input_ports;
        std::unordered_map<std::string, tt::CB> output_ports;

        std::string generate_reader_kernel() {
            std::stringstream ss;
            ss << "#include <cstdint>\n\n";
            ss << "void kernel_main() {\n";

            // Reader params from kernel args
            ss << "    uint32_t src_addr = get_arg_val<uint32_t>(0);\n";
            ss << "    uint32_t n_tiles = get_arg_val<uint32_t>(1);\n";
            ss << "    uint32_t start_tile = get_arg_val<uint32_t>(2);\n";

            // Circular buffers.
            ss << "\n";
            for (const auto& [name, cb] : input_ports) {
                ss << "    constexpr uint32_t " << name << " = " << static_cast<int>(cb) << ";\n";
            }
            ss << "\n";

            // Address generator.
            // TODO: Do we need this? How does this even work?
            ss << "    InterleavedAddrGenFast<true> a = {\n";
            ss << "        .bank_base_address = src_addr, \n";
            ss << "        .page_size = " << TILE_SIZE_BYTES << ", \n";
            ss << "        .data_format = DataFormat::Float16_b, \n";
            ss << "    };\n\n";

            // Tile stream loop.
            ss << "    for(uint32_t i = 0; i < n_tiles; i++) {\n";
            for (const auto& [name, cb] : input_ports) {
                ss << "        cb_reserve_back(" << name << ", 1);\n";
                ss << "        uint32_t " << name << "_addr = get_write_ptr(" << name << ");\n";
                ss << "        noc_async_read_tile(start_tile + i, a, " << name << "_addr);\n";
                ss << "        noc_async_read_barrier();\n";
                ss << "        cb_push_back(" << name << ", 1);\n";
            }
            ss << "    }\n";

            ss << "}\n";
            return ss.str();
        }
    };
}

std::shared_ptr<Buffer> MakeBuffer(Device *device, uint32_t size, uint32_t page_size, bool sram) {
    InterleavedBufferConfig config{
        .device= device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)
    };
    return CreateBuffer(config);
}

// Allocate a buffer on DRAM or SRAM. Assuming the buffer holds BFP16 data.
// A tile on Tenstorrent is 32x32 elements, given us using BFP16, we need 2 bytes per element.
// Making the tile size 32x32x2 = 2048 bytes.
// @param device: The device to allocate the buffer on.
// @param n_tiles: The number of tiles to allocate.
// @param sram: If true, allocate the buffer on SRAM, otherwise allocate it on DRAM.
std::shared_ptr<Buffer> MakeBufferBFP16(Device *device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    // For simplicity, all DRAM buffers have page size = tile size.
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(Program& program, const CoreSpec& core, tt::CB cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        size,
        {{
            cb,
            format
    }})
    .set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

// Circular buffers are Tenstorrent's way of communicating between the data movement and the compute kernels.
// kernels queue tiles into the circular buffer and takes them when they are ready. The circular buffer is
// backed by SRAM. There can be multiple circular buffers on a single Tensix core.
// @param program: The program to create the circular buffer on.
// @param core: The core to create the circular buffer on.
// @param cb: Which circular buffer to create (c_in0, c_in1, c_out0, c_out1, etc..). This is just an ID
// @param n_tiles: The number of tiles the circular buffer can hold.
CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CB cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
}

std::string next_arg(int& i, int argc, char **argv) {
    if(i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

void help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "This program demonstrates how to add two vectors using tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --device, -d <device_id>  Specify the device to run the program on. Default is 0.\n";
    std::cout << "  --seed, -s <seed>         Specify the seed for the random number generator. Default is random.\n";
    exit(0);
}

int main(int argc, char **argv) {
    int seed = std::random_device{}();
    int device_id = 0;

    // Quick and dirty argument parsing.
    for(int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if(arg == "--device" || arg == "-d") {
            device_id = std::stoi(next_arg(i, argc, argv));
        }
        else if(arg == "--seed" || arg == "-s") {
            seed = std::stoi(next_arg(i, argc, argv));
        }
        else if(arg == "--help" || arg == "-h") {
            help(argv[0]);
            return 0;
        }
        else {
            std::cout << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
        }
    }

    // Device and program setup.
    Device *device = CreateDevice(device_id);
    Program program = CreateProgram();
    CommandQueue& cq = device->command_queue();

    // Core grid setup.
    constexpr uint32_t num_cores = 1;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    tt::log_info("num_cores_x: {}, num_cores_y: {}", num_cores_x, num_cores_y);
    auto core_set = num_cores_to_corerange_set({0, 0}, num_cores, {num_cores_x, num_cores_y});
    tt::log_info("core_set: {}", core_set);
    tt::log_info("Total cores: {}", (*core_set.begin()).size());

    // Count determines how many tokens will be generated by our streams.
    uint32_t count = 1024 * 1;
    uint32_t n_tiles = std::ceil(count / TILE_SIZE);
    tt::log_info("count: {}, n_tiles: {}", count, n_tiles);

    // Divide tiles equally among cores.
    std::vector<uint32_t> tiles_per_core_vec(num_cores, n_tiles / num_cores);
    if (n_tiles % num_cores != 0) {
        tiles_per_core_vec[num_cores - 1] += n_tiles % num_cores;
    }
    std::cout << "Core work distribution: \n";
    for (uint32_t i = 0; i < num_cores; i++) {
        std::cout << "Core: " << i << " tiles: " << tiles_per_core_vec[i] << "\n";
    }
    std::cout << std::endl;

    // Input and output buffer setups.
    auto generator_buffer = MakeBufferBFP16(device, n_tiles, false);
    auto output_buffer = MakeBufferBFP16(device, n_tiles, false);
    std::mt19937 rng(seed);
    std::vector<uint32_t> generator_data = create_constant_vector_of_bfloat16(TILE_SIZE_BYTES * n_tiles, 0.0f);
    EnqueueWriteBuffer(cq, generator_buffer, generator_data, true);
    tt::log_info("Wrote generator buffer to DRAM");

    // Circular buffer setup.
    const uint32_t tiles_per_cb = 4;
    tt::log_info("tiles_per_cb: {}", tiles_per_cb);
    CBHandle cb_in = MakeCircularBufferBFP16(program, core_set, tt::CB::c_in0, tiles_per_cb);
    CBHandle cb_out = MakeCircularBufferBFP16(program, core_set, tt::CB::c_out0, tiles_per_cb);

    // Kernel generation.
    stream::Kernel reader_kernel;
    reader_kernel.add_input_port("cb_in0", tt::CB::c_in0);
    reader_kernel.generate_kernels();
    return 0;

    // A Tensix core is made up with 5 processors. 2 data movement processors, and 3 compute processors. The 2 data movement
    // processors act independent to other cores. And the 3 compute processors act together (hence 1 kerenl for compute).
    // There is no need to explicitly parallelize the compute kernels. Unlike traditional CPU/GPU style SPMD programming,
    // the 3 compute processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD engine), operates on the data, and
    // move it back to SRAM. The data movement processors moves data from the NoC, or in our case, the DRAM, into the SRAM.
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read` reads tiles from the input buffers A and B
    // into 2 circular buffers. `add` reads tiles from the circular buffers, adds them together, and dumps the result into
    // a third circular buffer. `tile_write` reads tiles from the third circular buffer and writes them to the output buffer C.
    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/stream/kernels/reader.cpp",
        core_set,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/stream/kernels/writer.cpp",
        core_set,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/stream/kernels/compute.cpp",
        core_set,
        ComputeConfig{
            .dst_full_sync_en = true, // TODO: What tf is this?
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    // Set runtime args for each core.
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        SetRuntimeArgs(program, reader, core, {
            generator_buffer->address(),
            tiles_per_core_vec[i],
            num_tiles_written,
        });
        SetRuntimeArgs(program, writer, core, {
            output_buffer->address(),
            tiles_per_core_vec[i],
            num_tiles_written
        });
        SetRuntimeArgs(program, compute, core, {
            tiles_per_core_vec[i],
        });

        num_tiles_written += tiles_per_core_vec[i];
    }


    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, true);
    auto end_time = std::chrono::high_resolution_clock::now();
    Finish(cq);

    // Calculate metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double latency_ms = duration.count() / 1000.0;  // Convert to milliseconds
    
    // Calculate data size and bandwidth
    size_t total_bytes = n_tiles * TILE_WIDTH * TILE_HEIGHT * sizeof(bfloat16);
    double bandwidth_mbps = (total_bytes) / (duration.count() * 1e-6) / 1e6;
    
    // Read the output buffer.
    std::vector<uint32_t> output_data;
    EnqueueReadBuffer(cq, output_buffer, output_data, true);

    // Print performance metrics
    std::cout << "\nPerformance Metrics:" << std::endl;
    std::cout << "Number of tiles: " << n_tiles << std::endl;
    std::cout << "Total # of cores: " << num_cores << std::endl;
    std::cout << "Total Execution Time: " << latency_ms << " ms" << std::endl;
    std::cout << "Latency per tile: " << latency_ms / n_tiles << " ms/tile" << std::endl;
    std::cout << "Total Bandwidth: " << bandwidth_mbps << " MB/s" << std::endl;
    std::cout << "Bandwidth per tile: " << bandwidth_mbps / n_tiles << " MB/s/tile" << std::endl;
    std::cout << "Data processed: " << total_bytes / 1024.0 / 1024.0 << " MB" << std::endl;

    // Finally, we close the device.
    CloseDevice(device);
    return 0;
}
