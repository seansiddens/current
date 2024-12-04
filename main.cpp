#include "bfloat8.hpp"
#include "common/core_coord.h"
#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "common/bfloat16.hpp"
#include "common/bfloat8.hpp"
#include "common/work_split.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include <chrono>
#include <iostream>

#include "common.hpp"
#include "stream.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

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
    // Device *device = CreateDevice(device_id);
    // Program program = CreateProgram();
    // CommandQueue& cq = device->command_queue();

    // Core grid setup.
    // constexpr uint32_t num_cores = 1;
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // uint32_t num_cores_x = compute_with_storage_grid_size.x;
    // uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // tt::log_info("num_cores_x: {}, num_cores_y: {}", num_cores_x, num_cores_y);
    // auto core_set = num_cores_to_corerange_set({0, 0}, num_cores, {num_cores_x, num_cores_y});
    // tt::log_info("core_set: {}", core_set);
    // tt::log_info("Total cores: {}", (*core_set.begin()).size());

    // Count determines how many tokens will be generated by our streams.
    uint32_t count = 1024 * 1024 * 512;
    auto type = tt::DataFormat::Float16_b;
    uint32_t n_tiles = std::ceil(count / TILE_SIZE);
    tt::log_info("count: {}, n_tiles: {}", count, n_tiles);

    // // Divide tiles equally among cores.
    // std::vector<uint32_t> tiles_per_core_vec(num_cores, n_tiles / num_cores);
    // if (n_tiles % num_cores != 0) {
    //     tiles_per_core_vec[num_cores - 1] += n_tiles % num_cores;
    // }
    // std::cout << "Core work distribution: \n";
    // for (uint32_t i = 0; i < num_cores; i++) {
    //     std::cout << "Core: " << i << " tiles: " << tiles_per_core_vec[i] << "\n";
    // }
    // std::cout << std::endl;

    // Input and output buffer setups.
    // auto generator_buffer = MakeBufferBFP16(device, n_tiles, false);
    // auto output_buffer = MakeBufferBFP16(device, n_tiles, false);
    std::mt19937 rng(seed);
    // std::vector<uint32_t> generator0_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 1.0f);
    std::vector<uint32_t> generator0_data = create_arange_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, false);
    // std::vector<uint32_t> generator_0_data = create_random_vector_of_bfp8(TILE_SIZE_BYTES * n_tiles, false, 10, seed);
    // std::vector<uint32_t> generator0_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 10.0f, seed);
    // std::vector<uint32_t> generator1_data = create_random_vector_of_bfp8(TILE_SIZE_BYTES * n_tiles, false, 10, seed);
    std::vector<uint32_t> generator1_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 2.0f);
    // std::vector<uint32_t> generator2_data = create_random_vector_of_bfp8(TILE_SIZE_BYTES * n_tiles, false, 10, seed);
    // std::vector<uint32_t> output_data = create_constant_vector_of_bfp8(TILE_SIZE_BYTES * n_tiles, 0.0f, false);
    std::vector<uint32_t> output_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 0.0f);
    // EnqueueWriteBuffer(cq, generator_buffer, generator_data, true);
    // tt::log_info("Wrote generator buffer to DRAM");

    // Circular buffer setup.
    // const uint32_t tiles_per_cb = 4;
    // tt::log_info("tiles_per_cb: {}", tiles_per_cb);
    // CBHandle cb_in = MakeCircularBufferBFP16(program, core_set, tt::CB::c_in0, tiles_per_cb);
    // CBHandle cb_out = MakeCircularBufferBFP16(program, core_set, tt::CB::c_out0, tiles_per_cb);

    // Kernel generation.
    current::Kernel kernel_a;
    current::Kernel kernel_b;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_input_port("in1", type);
    // kernel_a.add_input_port("in2", tt::DataFormat::Float16_b);
    kernel_a.add_output_port("out0", type);
    kernel_b.add_input_port("in0", type);
    kernel_b.add_output_port("out0", type);
    // Every implicitly has the input variable "inN" and result needs to be assigned to "outN".
    // The type of N is the index of the port.
    kernel_a.set_compute_kernel(R"(
        out0 = 2.0f * in0;
    )", true); // If matmul is TRUE, result of matmul will be in0. Computation will be applied elementwise.

    // kernel_b.set_compute_kernel(R"(
    //     out0 = in0;
    // )");

    // Define streams.
    current::Stream source0(generator0_data, count, type);
    current::Stream source1(generator1_data, count, type);
    // current::Stream source2(generator2_data, count, type);
    current::Stream sink(output_data, count, type);

    // Define connections between streams and kernels.
    // TODO: Segfaults if we add an extra kernel and don't connect it to anything.
    current::Map map({&kernel_a}, {&source0, &source1,  &sink});
    map.add_connection(&source0, &kernel_a, "in0");
    map.add_connection(&source1, &kernel_a, "in1");
    // map.add_connection(&source2, &kernel_a, "in2");
    map.add_connection(&kernel_a, "out0", &sink);
    // map.add_connection(&kernel_a, "out0", &kernel_b, "in0");
    // map.add_connection(&kernel_b, "out0", &sink);
    // map.export_dot("stream_graph.dot");
    // map.generate_device_kernels();
    // map.check_connections();
    // map.generate_device_kernels();
    map.execute();

    auto out = map.read_stream(&sink);
    // std::vector<bfloat16> output_bf16 = unpack_uint32_vec_into_bfloat16_vec(out);
    // for (uint32_t i = 0; i < output_bf16.size(); i++) {
    //     std::cout << i << ": " << output_bf16[i].to_float() << "\n";
    // }
    assert(out == output_data);

    std::cout << std::endl;

    return 0;
}
