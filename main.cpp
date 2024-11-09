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

        uint32_t num_input_ports() const {
            return input_ports.size();
        }

        uint32_t num_output_ports() const {
            return output_ports.size();
        }

        // TODO: For each input port, we need need a CB to move data from NOC to compute.
        // If the kernel is a generator, each input port will be reading from a DRAM buffer.
        // If our input/output ports are connected to other kernels, need to determine how to do
        // the pipelining.
        std::unordered_map<std::string, tt::CB> input_ports;
        std::unordered_map<std::string, tt::CB> output_ports;
    };

    class Map {
      public:
        Map(std::vector<Kernel *> kernels) : kernels(kernels) {
            // Allocate port map.
            port_map.resize(kernels.size());
            for (size_t i = 0; i < kernels.size(); i++) {
                port_map[i].resize(kernels.size());
            }

            std::cout << "port_map dimensions: " << port_map.size() << " x " << port_map[0].size() << std::endl;
        }

        void add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in) {
            // TODO: Test this.
            // Get indices of src and dst.
            size_t src_idx = std::find(kernels.begin(), kernels.end(), src) - kernels.begin();
            size_t dst_idx = std::find(kernels.begin(), kernels.end(), dst) - kernels.begin();
            if (src_idx >= kernels.size() || dst_idx >= kernels.size()) {
                std::cerr << "Failed to add kernel connection!\n";
                exit(1);
            }
            port_map[src_idx][dst_idx] = std::make_pair(src_out, dst_in);
        }

        void execute() {
            generate_device_kernels();
        }

        void generate_device_kernels() {
            for (int i = 0; i < kernels.size(); i++) {
                Kernel *kernel = kernels[i];
                auto input_ports = kernel->input_ports;
                auto output_ports = kernel->output_ports;
                auto num_input_ports = kernel->num_input_ports();
                auto num_output_ports = kernel->num_output_ports();
                
                // Generate reader kernel.
                std::stringstream rs;
                rs << "#include <cstdint>\n\n";
                rs << "void kernel_main() {\n";

                // Reader params from kernel args
                // TODO: If our input ports are not connected to other kernels, we are reading from a DRAM buffer.
                rs << "    uint32_t src_addr = get_arg_val<uint32_t>(0);\n";
                rs << "    uint32_t n_tiles = get_arg_val<uint32_t>(1);\n";
                rs << "    uint32_t start_tile = get_arg_val<uint32_t>(2);\n";

                // Circular buffers.
                rs << "\n";
                for (const auto& [name, cb] : input_ports) {
                    rs << "    constexpr uint32_t " << name << " = " << static_cast<int>(cb) << ";\n";
                }
                rs << "\n";

                // Address generator.
                // TODO: Do we need this? How does this even work?
                rs << "    InterleavedAddrGenFast<true> a = {\n";
                rs << "        .bank_base_address = src_addr, \n";
                rs << "        .page_size = " << TILE_SIZE_BYTES << ", \n";
                rs << "        .data_format = DataFormat::Float16_b, \n";
                rs << "    };\n\n";

                // Tile stream loop.
                rs << "    for(uint32_t i = 0; i < n_tiles; i++) {\n";
                for (const auto& [name, cb] : input_ports) {
                    rs << "        cb_reserve_back(" << name << ", 1);\n";
                    rs << "        uint32_t " << name << "_addr = get_write_ptr(" << name << ");\n";
                    rs << "        noc_async_read_tile(start_tile + i, a, " << name << "_addr);\n";
                    rs << "        noc_async_read_barrier();\n";
                    rs << "        cb_push_back(" << name << ", 1);\n";
                }
                rs << "    }\n";
                rs << "}\n";

                const std::string generated_reader_kernel_path = "tt_metal/programming_examples/personal/stream/kernels/generated/reader.cpp";
                auto reader_kernel_file = std::ofstream(generated_reader_kernel_path);
                if (!reader_kernel_file.is_open()) {
                    tt::log_error("Failed to open file for writing: {}", generated_reader_kernel_path);
                }
                reader_kernel_file << rs.str();
                reader_kernel_file.close();

                // Generate compute kernel.
                std::stringstream cs;
                // Includes.
                cs << "#include \"compute_kernel_api/common.h\"\n";
                cs << "#include \"compute_kernel_api/tile_move_copy.h\"\n";
                cs << "#include \"compute_kernel_api/eltwise_binary.h\"\n";
                cs << "#include \"compute_kernel_api/eltwise_unary/eltwise_unary.h\"\n";
                cs << "#include \"compute_kernel_api.h\"\n";
                cs << "#include \"sfpi.h\"\n";
                cs << "#include \"debug/dprint.h\"\n";
                cs << "\n";

                // SFPU computation
                // TODO: Need to somehow parameterize this compute source code generation.
                cs << "namespace sfpi {\n";
                cs << "template< int ITERATIONS = 16 >\n";
                cs << "sfpi_inline void compute() {\n";
                cs << "    for (int i = 0; i < ITERATIONS; i++) {\n";
                cs << "        vFloat in = dst_reg[i];\n";
                cs << "        vFloat a = in + 1.0f;\n";
                cs << "        vFloat out = a;\n";
                cs << "        dst_reg[i] = out;\n";
                cs << "    }\n";
                cs << "}\n";
                cs << "}\n";
                cs << "\n";

                // Main function.
                cs << "namespace NAMESPACE {\n";
                cs << "void MAIN {\n";

                // Get kernel args.
                // TODO: How will this be arbitrarily determined by the runtime?
                // If each generator stream has a static count, then I assume this can be figured out.
                cs << "    uint32_t n_tiles = get_arg_val<uint32_t>(0);\n";

                // Circular buffers.
                // TODO: Do we have a CB for each input port and each output port? 
                for (const auto& [name, cb] : input_ports) {
                    cs << "    constexpr uint32_t " << name << " = " << static_cast<int>(cb) << ";\n";
                }
                for (const auto& [name, cb] : output_ports) {
                    cs << "    constexpr uint32_t " << name << " = " << static_cast<int>(cb) << ";\n";
                }
                cs << "\n";

                // Initialize SFPU.
                // TODO: Init state needs to be setup for every input CB.
                cs << "    init_sfpu(" << input_ports.begin()->first<< ");\n";
                cs << "\n";

                // Tile stream loop
                cs << "    for(uint32_t i = 0; i < n_tiles; i++) {\n";
                cs << "        cb_wait_front(cb_in0, 1);\n";
                cs << "        copy_tile(cb_in0, 0, 0);\n";
                cs << "        tile_regs_acquire();\n";
                cs << "        MATH((sfpi::compute()));\n";
                cs << "        tile_regs_commit();\n";
                cs << "        cb_pop_front(cb_in0, 1);\n";
                cs << "        tile_regs_wait();\n";
                cs << "        cb_reserve_back(cb_out0, 1);\n";
                cs << "        pack_tile(0, cb_out0);\n";
                cs << "        cb_push_back(cb_out0, 1);\n";
                cs << "        tile_regs_release();\n";
                // End tile stream loop.
                cs << "    }\n";

                // End main.
                cs << "}\n";
                cs << "}\n";
                cs << "\n";

                // Save to file.
                const std::string generated_compute_kernel_path = "tt_metal/programming_examples/personal/stream/kernels/generated/compute.cpp";
                auto compute_kernel_file = std::ofstream(generated_compute_kernel_path);
                if (!compute_kernel_file.is_open()) {
                    tt::log_error("Failed to open file for writing: {}", generated_compute_kernel_path);
                }
                compute_kernel_file << cs.str();
                compute_kernel_file.close();

                std::stringstream ws;
                // Includes.
                ws << "#include <cstdint>\n";
                ws << "\n";

                // Main 
                ws << "void kernel_main() {\n";

                // Get kernel runtime args.
                ws << "    uint32_t dst_addr = get_arg_val<uint32_t>(0);\n";
                ws << "    uint32_t n_tiles = get_arg_val<uint32_t>(1);\n";
                ws << "    uint32_t start_tile = get_arg_val<uint32_t>(2);\n";
                ws << "\n";

                // Circular buffers.
                for (const auto& [name, cb] : output_ports) {
                    ws << "    constexpr uint32_t " << name << " = " << static_cast<int>(cb) << ";\n";
                }
                ws << "\n";

                // Address generator.
                // TODO: Do we need this? How does this even work?
                ws << "    InterleavedAddrGenFast<true> c = {\n";
                ws << "        .bank_base_address = dst_addr, \n";
                ws << "        .page_size = " << TILE_SIZE_BYTES << ", \n";
                ws << "        .data_format = DataFormat::Float16_b, \n";
                ws << "    };\n\n";

                // Tile stream loop.
                ws << "    for(uint32_t i = 0; i < n_tiles; i++) {\n";
                ws << "        cb_wait_front(cb_out0, 1);\n";
                ws << "        uint32_t cb_out0_addr = get_read_ptr(cb_out0);\n";
                ws << "        noc_async_write_tile(start_tile + i, c, cb_out0_addr);\n";
                ws << "        noc_async_write_barrier();\n";
                // TODO: Potentially slower than just using noc_async_write_flushed().
                // Might not even have to use until the last tile is written.
                ws << "        cb_pop_front(cb_out0, 1);\n";

                // End tile stream loop
                ws << "    }\n";

                //End Main
                ws << "}\n";
                ws << "\n";

                // Save to file.
                const std::string generated_writer_kernel_path = "tt_metal/programming_examples/personal/stream/kernels/generated/writer.cpp";
                auto writer_kernel_file = std::ofstream(generated_writer_kernel_path);
                if (!writer_kernel_file.is_open()) {
                    tt::log_error("Failed to open file for writing: {}", generated_writer_kernel_path);
                }
                writer_kernel_file << ws.str();
                writer_kernel_file.close();
            }
        }

      private:
        std::vector<Kernel *> kernels;
        // Adjaceny matrix representing our kernel flow graph.
        // Entry at [i][j] represents that we have an outgoing stream from kernel[i] to kernel[j].
        // The entry stores a pair <out_port, in_port> representing a connection from kernel[i]'s out_port to kernel[j]'s in_port.
        // TODO: Right now we are only storing a single pair for each edge. This means we can only have one connection between any two kernels.
        std::vector<std::vector<std::pair<std::string, std::string>>> port_map;
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
    reader_kernel.add_output_port("cb_out0", tt::CB::c_out0);

    stream::Map map({&reader_kernel});
    map.generate_device_kernels();

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/stream/kernels/generated/reader.cpp",
        core_set,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/stream/kernels/generated/writer.cpp",
        core_set,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/stream/kernels/generated/compute.cpp",
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

    // Print output data.
    std::vector<bfloat16> output = unpack_uint32_vec_into_bfloat16_vec(output_data);
    for (uint32_t i = 0; i < output.size(); i++) {
        std::cout << output[i].to_float() << " ";
    }
    std::cout << std::endl;

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
