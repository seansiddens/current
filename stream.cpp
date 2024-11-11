#include "stream.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "impl/buffers/buffer.hpp"
#include "work_split.hpp"

#include "common.hpp"

namespace stream {


// Add a new input or output port to the kernel
void Kernel::add_input_port(const std::string& name, tt::CB cb)  {
    input_ports[name] = cb;
}

void Kernel::add_output_port(const std::string& name, tt::CB cb) {
    output_ports[name] = cb;
}

uint32_t Kernel::num_input_ports() const {
    return input_ports.size();
}

uint32_t Kernel::num_output_ports() const {
    return output_ports.size();
}

Map::Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams) : kernels(kernels), streams(streams) {}

void Map::add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in) {
    // TODO: Add error handling.
    Endpoint src_endpoint = {Endpoint::Type::Kernel, get_kernel_index(src), src_out};
    Endpoint dst_endpoint = {Endpoint::Type::Kernel, get_kernel_index(dst), dst_in};
    add_connection(src_endpoint, dst_endpoint);
}

void Map::add_connection(Stream *src, Kernel *dst, std::string dst_in) {
    Endpoint src_endpoint = {Endpoint::Type::Stream, get_stream_index(src), ""};
    Endpoint dst_endpoint = {Endpoint::Type::Kernel, get_kernel_index(dst), dst_in};
    add_connection(src_endpoint, dst_endpoint);
}

void Map::add_connection(Kernel *src, std::string src_out, Stream *dst) {
    Endpoint src_endpoint = {Endpoint::Type::Kernel, get_kernel_index(src), src_out};
    Endpoint dst_endpoint = {Endpoint::Type::Stream, get_stream_index(dst), ""};
    // TODO: Need to do checks whether these are valid port names and that the ports have not already been connected.
    add_connection(src_endpoint, dst_endpoint);
}

void Map::execute() {
    // 1. Create device and program.
    runtime.device = tt_metal::CreateDevice(0);
    if (!runtime.device) {
        std::cerr << "Failed to create device!\n";
        exit(1);
    }
    runtime.program = tt_metal::CreateProgram();

    // 2. Core grid setup.
    // TODO: Have this configurable by user and dyanmic by runtime scheduling.
    runtime.num_cores = 1;
    auto compute_with_storage_grid_size = runtime.device->compute_with_storage_grid_size();
    runtime.num_cores_x = compute_with_storage_grid_size.x;
    runtime.num_cores_y = compute_with_storage_grid_size.y;
    runtime.core_set = num_cores_to_corerange_set({0, 0}, runtime.num_cores, {runtime.num_cores_x, runtime.num_cores_y});
    tt::log_info("num_cores_x: {}, num_cores_y: {}", runtime.num_cores_x, runtime.num_cores_y);
    tt::log_info("core_set: {}", runtime.core_set);
    tt::log_info("Total cores: {}", (*runtime.core_set.begin()).size());

    // 3. Input & Output DRAM buffer setup.
    for (size_t i = 0; i < streams.size(); i++) {
        auto stream = streams[i];
        tt_metal::InterleavedBufferConfig config = {
            .device = runtime.device,
            .size = stream->n_elements * stream->element_size,
            .page_size = stream->element_size * TILE_WIDTH * TILE_HEIGHT, // TODO: Not sure what is optimal for this.
            .buffer_type = tt_metal::BufferType::DRAM
        };
        stream->device_buffer = tt_metal::CreateBuffer(config);
        // TODO: Does this need to be blocking?
        // TODO: What if there's a mismatch between the host data size and the device buffer size?
        tt_metal::EnqueueWriteBuffer(runtime.device->command_queue(), stream->device_buffer, stream->host_data, true);
        stream->device_buffer_address = stream->device_buffer->address();
        stream->device_buffer_noc_coordinates = stream->device_buffer->noc_coordinates();
    }

    // 4. Generate device kernels.
    generate_device_kernels();
}

bool Map::has_incoming_connection(Kernel *kernel) {
    // Get the index of our kernel in the kernels vector
    size_t kernel_idx = get_kernel_index(kernel);
    
    // Check each connection
    for (const Connection& connection : connections) {
        // Is the destination endpoint a kernel?
        if (connection.dest.is_kernel()) {
            // Does its index match our target kernel?
            if (connection.dest.index == kernel_idx) {
                return true;  // Found an incoming connection!
            }
        }
    }
    return false;
}

void Map::generate_reader_device_kernel(Kernel *kernel) {
    auto input_ports = kernel->input_ports;
    auto output_ports = kernel->output_ports;
    auto num_input_ports = kernel->num_input_ports();
    auto num_output_ports = kernel->num_output_ports();

    // Generate reader kernel.
    std::stringstream rs;

    // Check whether this kernel has an incoming connection.
    auto has_incoming_connection = Map::has_incoming_connection(kernel);
    if (!has_incoming_connection) {
        std::cout << "Kernel has no incoming connection!\n";
        // This kernel has no incoming connections, so it is simply reading from a DRAM buffer.

        rs << "#include <cstdint>\n\n";
        rs << "void kernel_main() {\n";

        // Reader params from kernel args
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
            // TODO: Could probably optimize this by batching reads/loads to CB.
            rs << "        cb_reserve_back(" << name << ", 1);\n";
            rs << "        uint32_t " << name << "_addr = get_write_ptr(" << name << ");\n";
            rs << "        noc_async_read_tile(start_tile + i, a, " << name << "_addr);\n";
            rs << "        noc_async_read_barrier();\n";
            rs << "        cb_push_back(" << name << ", 1);\n";
        }
        rs << "    }\n";
        rs << "}\n";

    } else {
        std::cout << "Kernel has incoming connection!\n";
    }


    const std::string generated_reader_kernel_path = "tt_metal/programming_examples/personal/stream/kernels/generated/reader.cpp";
    auto reader_kernel_file = std::ofstream(generated_reader_kernel_path);
    if (!reader_kernel_file.is_open()) {
        tt::log_error("Failed to open file for writing: {}", generated_reader_kernel_path);
    }
    reader_kernel_file << rs.str();
    reader_kernel_file.close();
    std::cout << "Generated reader kernel!\n";
}

void Map::generate_compute_device_kernel(Kernel *kernel) {
    auto input_ports = kernel->input_ports;
    auto output_ports = kernel->output_ports;
    auto num_input_ports = kernel->num_input_ports();
    auto num_output_ports = kernel->num_output_ports();

    // Generate compute kernel.
    std::stringstream cs; // Includes.
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
    // TODO: Need to handle multiple input CBs.
    cs << "        cb_wait_front(" << input_ports.begin()->first << ", 1);\n";
    cs << "        copy_tile(" << input_ports.begin()->first << ", 0, 0);\n";
    cs << "        tile_regs_acquire();\n";
    cs << "        MATH((sfpi::compute()));\n";
    cs << "        tile_regs_commit();\n";
    cs << "        cb_pop_front(" << input_ports.begin()->first << ", 1);\n";
    cs << "        tile_regs_wait();\n";
    cs << "        cb_reserve_back(" << output_ports.begin()->first << ", 1);\n";
    cs << "        pack_tile(0, " << output_ports.begin()->first << ");\n";
    cs << "        cb_push_back(" << output_ports.begin()->first << ", 1);\n";
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
}

void Map::generate_writer_device_kernel(Kernel *kernel) {
    auto input_ports = kernel->input_ports;
    auto output_ports = kernel->output_ports;
    auto num_input_ports = kernel->num_input_ports();
    auto num_output_ports = kernel->num_output_ports();

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
    // TODO: Handle multiple output CBs.
    ws << "        cb_wait_front(" << output_ports.begin()->first << ", 1);\n";
    ws << "        uint32_t cb_out0_addr = get_read_ptr(" << output_ports.begin()->first << ");\n";
    ws << "        noc_async_write_tile(start_tile + i, c, cb_out0_addr);\n";
    ws << "        noc_async_write_barrier();\n";
    // TODO: Potentially slower than just using noc_async_write_flushed().
    // Might not even have to use until the last tile is written.
    ws << "        cb_pop_front(" << output_ports.begin()->first << ", 1);\n";

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

void Map::generate_device_kernels() {
    for (int i = 0; i < kernels.size(); i++) {
        Kernel *kernel = kernels[i];
        generate_reader_device_kernel(kernel);
        generate_compute_device_kernel(kernel);
        generate_writer_device_kernel(kernel);
    }
}

} // End namespace stream