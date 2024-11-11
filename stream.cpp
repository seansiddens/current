#include "stream.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "current/common.hpp"
#include "impl/buffers/buffer.hpp"
#include "work_split.hpp"

namespace stream {


// Add a new input or output port to the kernel
void Kernel::add_input_port(const std::string& name, tt::DataFormat data_format)  {
    input_ports.push_back({name, data_format});
}

void Kernel::add_output_port(const std::string& name, tt::DataFormat data_format) {
    output_ports.push_back({name, data_format});
}

uint32_t Kernel::num_input_ports() const {
    return input_ports.size();
}

uint32_t Kernel::num_output_ports() const {
    return output_ports.size();
}

Map::Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams) : kernels(kernels), streams(streams) {}

// TODO: Validate that port connections are valid (check that types are the same.)
void Map::add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in) {
    // TODO: Add error handling.
    Endpoint src_endpoint = {Endpoint::EndpointType::Kernel, get_kernel_index(src), src_out};
    Endpoint dst_endpoint = {Endpoint::EndpointType::Kernel, get_kernel_index(dst), dst_in};
    add_connection(src_endpoint, dst_endpoint);
}

void Map::add_connection(Stream *src, Kernel *dst, std::string dst_in) {
    Endpoint src_endpoint = {Endpoint::EndpointType::Stream, get_stream_index(src), ""};
    Endpoint dst_endpoint = {Endpoint::EndpointType::Kernel, get_kernel_index(dst), dst_in};
    add_connection(src_endpoint, dst_endpoint);
}

void Map::add_connection(Kernel *src, std::string src_out, Stream *dst) {
    Endpoint src_endpoint = {Endpoint::EndpointType::Kernel, get_kernel_index(src), src_out};
    Endpoint dst_endpoint = {Endpoint::EndpointType::Stream, get_stream_index(dst), ""};
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

    // TODO: CBs are pinned to specific cores, need to figure out kernel placement before this.
    // // For every kernel in our program, we will have a CB for each input and output port.
    // for (size_t i = 0; i < kernels.size(); i++) {
    //     auto kernel = kernels[i];
    //     for (const auto& input_port : kernel->input_ports) {
    //         tt::CircularBufferConfig cb_config = CircularBufferConfig(
    //             .size = TILES_PER_CB,
    //         )
    //     }
    // }

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

std::vector<Map::Connection> Map::get_incoming_connections(Kernel *kernel) {
    // Get the index of our kernel in the kernels vector
    size_t kernel_idx = get_kernel_index(kernel);

    // Find all connections that have our kernel as the destination
    std::vector<Connection> incoming_connections;
    for (const Connection& connection : connections) {
        if (connection.dest.index == kernel_idx) {
            incoming_connections.push_back(connection);
        }
    }
    return incoming_connections;
}

std::vector<Map::Connection> Map::get_outgoing_connections(Kernel *kernel) {
    // Get the index of our kernel in the kernels vector
    size_t kernel_idx = get_kernel_index(kernel);

    // Find all connections that have our kernel as the source
    std::vector<Connection> outgoing_connections;
    for (const Connection& connection : connections) {
        if (connection.source.index == kernel_idx) {
            outgoing_connections.push_back(connection);
        }
    }
    return outgoing_connections;
}

std::string data_format_to_string(tt::DataFormat data_format) {
    // std::cout << "Data format: " << data_format << "\n";
    switch (data_format) {
        case tt::DataFormat::Float16_b: return "DataFormat::Float16_b";
        default:
            std::cerr << "Unsupported data format!\n";
            exit(1);
    }
}

void Map::generate_reader_device_kernel(
    Kernel *kernel,
    std::vector<Kernel::Port> input_ports,
    std::vector<Connection> incoming_connections
) {

    // Generate reader kernel.
    std::stringstream rs;
    rs << "#include <cstdint>\n";
    rs << "#include \"dataflow_api.h\"\n";
    rs << "void kernel_main() {\n";

    // Reader params from kernel args
    uint32_t total_args = 0;

    for (size_t i = 0; i < incoming_connections.size(); i++) {
        auto connection = incoming_connections[i];
        if (connection.source.is_stream()) {
            auto stream = streams[connection.source.index];
            // Total # of tiles this kernel will read from this stream.
            // Stream -> Kernel, get the input port index.
            auto port_index = kernel->get_input_port_index(connection.dest.port);
            rs << "    uint32_t source" << port_index << "_n_tiles = get_arg_val<uint32_t>(" << total_args << ");\n";
            total_args++;
            // For every incoming stream connection, we need to get it's address and create an address generator.
            rs << "    uint32_t source" << port_index << "_addr = get_arg_val<uint32_t>(" << total_args << ");\n";
            total_args++;
            // Address generator.
            // TODO: Do we need this? How does this even work?
            rs << "    const InterleavedAddrGenFast<true> source" << port_index << "_addr_gen = {\n";
            rs << "        .bank_base_address = source" << port_index << "_addr, \n";
            rs << "        .page_size = " << TILE_WIDTH * TILE_HEIGHT * stream->element_size << ", \n";
            rs << "        .data_format = " << data_format_to_string(stream->data_format) << ", \n";
            rs << "    };\n\n";
        } else {
            // TODO: Handle incoming kernel connections.
        }
    }

    // Circular buffers.
    uint32_t num_input_cbs = 0;
    for (size_t i = 0; i < input_ports.size(); i++) {
        // Assign CBs to input ports in iteration order.
        auto port = input_ports[i];
        rs << "    constexpr uint32_t " << port.name << " = " << num_input_cbs << ";\n";
        num_input_cbs++;
    }
    rs << "\n";

    // Input tile stream loop.
    // TODO: Handle multiple input ports with DIFFERENT n_tiles.
    // In the loop, need to keep track of how many tiles we've read from each input port.
    // Break condition is when we've read the expected # of tiles from each input port.
    rs << "    for(uint32_t i = 0; i < source0_n_tiles; i++) {\n";
    // Wait for space in CBs
    for (size_t i = 0; i < input_ports.size(); i++) {
        rs << "        cb_reserve_back(" << input_ports[i].name << ", 1);\n";
    }
    // Read tile into CB from DRAM.
    for (size_t i = 0; i < input_ports.size(); i++) {
        rs << "        uint32_t " << input_ports[i].name << "_write_ptr = get_write_ptr(" << input_ports[i].name << ");\n";
        rs << "        noc_async_read_tile(i, source" << i << "_addr_gen, " << input_ports[i].name << "_write_ptr);\n";
    }
    // Wait until tile reads are done.
    rs << "\n";
    rs << "        noc_async_read_barrier();\n";

    // Push tiles into CBs
    // Signals to compute engine that a tile is ready to be processed.
    for (size_t i = 0; i < input_ports.size(); i++) {
        rs << "        cb_push_back(" << input_ports[i].name << ", 1);\n";
    }

    rs << "    }\n";
    // End tile stream loop.

    rs << "}\n";
    rs << "\n";

    const std::string generated_reader_kernel_path = "tt_metal/programming_examples/personal/current/kernels/generated/reader.cpp";
    auto reader_kernel_file = std::ofstream(generated_reader_kernel_path);
    if (!reader_kernel_file.is_open()) {
        tt::log_error("Failed to open file for writing: {}", generated_reader_kernel_path);
    }
    reader_kernel_file << rs.str();
    reader_kernel_file.close();
    std::cout << "Generated reader kernel!\n";
}

void Map::generate_compute_device_kernel(
    Kernel *kernel,
    std::vector<Kernel::Port> input_ports,
    std::vector<Kernel::Port> output_ports
) {
    std::stringstream cs; 
    // Includes
    cs << "#include \"compute_kernel_api/common.h\"\n";
    cs << "#include \"compute_kernel_api/tile_move_copy.h\"\n";
    cs << "#include \"compute_kernel_api/eltwise_binary.h\"\n";
    cs << "#include \"compute_kernel_api/eltwise_unary/eltwise_unary.h\"\n";
    cs << "#include \"compute_kernel_api.h\"\n";
    cs << "#include \"sfpi.h\"\n";
    // cs << "#include \"debug/dprint.h\"\n";
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
    // TODO: Have varying # of tiles for each input port.
    uint32_t total_args = 0;
    cs << "    uint32_t n_tiles = get_arg_val<uint32_t>(" << total_args << ");\n";
    total_args++;
    cs << "\n";

    // CBs we are going to use.
    // TODO: Right now input/output ports just correspond to input/output CBs.
    // I think TT supports arbitrary usage of CBs, so could be more flexible for how we do this assignment.
    // e.g letting us have more ports or use them more efficiently.
    // We also might end up using intermediate CBs at some point for computation.
    uint32_t num_input_cbs = IN_CB_START;
    for (size_t i = 0; i < input_ports.size(); i++) {
        // Assign CBs to input ports in iteration order.
        auto port = input_ports[i];
        cs << "    constexpr uint32_t " << port.name << " = " << num_input_cbs << ";\n";
        num_input_cbs++;
    }
    cs << "\n";
    uint32_t num_output_cbs = OUT_CB_START; // Output CBs start at index 16.
    for (size_t i = 0; i < output_ports.size(); i++) {
        // Assign CBs to output ports in iteration order.
        auto port = output_ports[i];
        cs << "    constexpr uint32_t " << port.name << " = " << num_output_cbs << ";\n";
        num_output_cbs++;
    }
    cs << "\n";



    // Initialize SFPU.
    for (size_t i = 0; i < input_ports.size(); i++) {
        cs << "    init_sfpu(" << input_ports[i].name << ");\n";
    }
    cs << "\n";

    // Tile stream loop
    cs << "    for(uint32_t i = 0; i < n_tiles; i++) {\n";
    // Wait for tiles to be read in CBs.
    for (size_t i = 0; i < input_ports.size(); i++) {
        cs << "        cb_wait_front(" << input_ports[i].name << ", 1);\n";
    }
    cs << "\n";

    // Copy tiles from CBs to SFPU registers.
    for (size_t i = 0; i < input_ports.size(); i++) {
        /**
        * Copies a single tile from the specified input CB and writes the result to
        * DST at a specified index. The function will employ unpacker to first unpack into SRC
        * registers and then perform move into DST registers, at a specified index.
        * For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
        * previously called to ensure that at least some number n>0 of tiles are available
        * in the input CB. The CB index 0 then references the first tile in the received section of the CB,
        * up to index n-1 (in a FIFO order). The DST register buffer must be in acquired state via
        * acquire_dst call. This call is blocking and is only available on the compute
        * engine.
        *
        * Return value: None
        *
        * | Argument       | Description                                       | Data type | Valid range                                         | required |
        * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
        * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | Yes      |
        * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | Yes      |
        * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | Yes      |
        * */
        cs << "        copy_tile(" << input_ports[i].name << ", 0, " << i << ");\n";
    }
    cs << "\n";
    cs << "        tile_regs_acquire();\n";
    cs << "        MATH((sfpi::compute()));\n";
    cs << "        tile_regs_commit();\n";
    cs << "\n";
    // Computation finished, pop tiles from input CBs
    // TODO: This may be able to be re-ordered?
    for (size_t i = 0; i < input_ports.size(); i++) {
        cs << "        cb_pop_front(" << input_ports[i].name << ", 1);\n";
    }
    cs << "\n";

    // Packer waits here until the SFPU is done.
    cs << "        tile_regs_wait();\n";
    // Reserve space in output CBs.
    for (size_t i = 0; i < output_ports.size(); i++) {
        cs << "        cb_reserve_back(" << output_ports[i].name << ", 1);\n";
    }
    cs << "\n";
    // Pack tiles into output CBs.
    for (size_t i = 0; i < output_ports.size(); i++) {
        cs << "        pack_tile(" << i << ", " << output_ports[i].name << ");\n";
    }
    cs << "\n";
    // Announce that the output tiles are ready.
    for (size_t i = 0; i < output_ports.size(); i++) {
        cs << "        cb_push_back(" << output_ports[i].name << ", 1);\n";
    }
    cs << "\n";

    // Packer releases the SFPU registers.
    cs << "        tile_regs_release();\n";

    // End tile stream loop.
    cs << "    }\n";

    // End main.
    cs << "}\n";
    cs << "}\n";
    cs << "\n";

    // Save to file.
    const std::string generated_compute_kernel_path = "tt_metal/programming_examples/personal/current/kernels/generated/compute.cpp";
    auto compute_kernel_file = std::ofstream(generated_compute_kernel_path);
    if (!compute_kernel_file.is_open()) {
        tt::log_error("Failed to open file for writing: {}", generated_compute_kernel_path);
    }
    compute_kernel_file << cs.str();
    compute_kernel_file.close();
}

void Map::generate_writer_device_kernel(
    Kernel *kernel,
    std::vector<Kernel::Port> output_ports,
    std::vector<Connection> outgoing_connections
) {

    std::stringstream ws;
    // Includes.
    ws << "#include <cstdint>\n";
    ws << "\n";

    // Main 
    ws << "void kernel_main() {\n";

    // Writer params from kernel args
    uint32_t total_args = 0;
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (connection.dest.is_stream()) {
            auto stream = streams[connection.dest.index];
            // Kernel -> Stream, get the output port index.
            auto port_index = kernel->get_output_port_index(connection.source.port);
            // Total # of tiles this kernel will write to this stream.
            ws << "    uint32_t sink" << port_index << "_n_tiles = get_arg_val<uint32_t>(" << total_args << ");\n";
            total_args++;
            // For every outgoing stream connection, we need to get it's address and create an address generator.
            ws << "    uint32_t sink" << port_index << "_addr = get_arg_val<uint32_t>(" << total_args << ");\n";
            total_args++;
            // Address generator.
            // TODO: Do we need this? How does this even work?
            ws << "    const InterleavedAddrGenFast<true> sink" << port_index << "_addr_gen = {\n";
            ws << "        .bank_base_address = sink" << port_index << "_addr, \n";
            ws << "        .page_size = " << TILE_WIDTH * TILE_HEIGHT * stream->element_size << ", \n";
            ws << "        .data_format = " << data_format_to_string(stream->data_format) << ", \n";
            ws << "    };\n\n";
        } else {
            // TODO: Handle outgoing kernel connections.
        }
    }

    // Circular buffers.
    uint32_t num_output_cbs = OUT_CB_START; // Output CBs start at index 16.
    for (size_t i = 0; i < output_ports.size(); i++) {
        // Assign CBs to input ports in iteration order.
        auto port = output_ports[i];
        ws << "    constexpr uint32_t " << port.name << " = " << num_output_cbs << ";\n";
        num_output_cbs++;
    }
    ws << "\n";

    // Output tile stream loop.
    // TODO: Handle multiple output ports with DIFFERENT n_tiles.
    // In the loop, need to keep track of how many tiles we've written to each output port.
    // Break condition is when we've written the expected # of tiles to each output port.
    ws << "    for(uint32_t i = 0; i < sink0_n_tiles; i++) {\n";
    // Wait tiles to arrive in CBs
    for (size_t i = 0; i < output_ports.size(); i++) {
        ws << "        cb_wait_front(" << output_ports[i].name << ", 1);\n";
    }

    // Write tiles to DRAM.
    for (size_t i = 0; i < output_ports.size(); i++) {
        ws << "        uint32_t " << output_ports[i].name << "_read_ptr = get_read_ptr(" << output_ports[i].name << ");\n";
        ws << "        noc_async_write_tile(i, sink" << i << "_addr_gen, " << output_ports[i].name << "_read_ptr);\n";
    }
    // Wait until tile writes are done.
    ws << "\n";
    ws << "        noc_async_write_barrier();\n";
    // // TODO: Potentially slower than just using noc_async_write_flushed().
    // // Might not even have to use until the last tile is written.

    // Mark the tiles as consumed.
    for (size_t i = 0; i < output_ports.size(); i++) {
        ws << "        cb_pop_front(" << output_ports[i].name << ", 1);\n";
    }
    ws << "    }\n";
    // End tile stream loop.

    //End Main
    ws << "}\n";
    ws << "\n";

    // Save to file.
    const std::string generated_writer_kernel_path = "tt_metal/programming_examples/personal/current/kernels/generated/writer.cpp";
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
        auto input_ports = kernel->input_ports;
        auto output_ports = kernel->output_ports;
        auto num_input_ports = kernel->num_input_ports();
        auto num_output_ports = kernel->num_output_ports();
        auto incoming_connections = get_incoming_connections(kernel);
        auto outgoing_connections = get_outgoing_connections(kernel);

        generate_reader_device_kernel(kernel, input_ports, incoming_connections);
        generate_compute_device_kernel(kernel, input_ports, output_ports);
        generate_writer_device_kernel(kernel, output_ports, outgoing_connections);
    }
}

void Map::export_dot(const std::string& filename) const {
    std::ofstream dot_file(filename);
    
    // Header
    dot_file << "digraph StreamGraph {\n";
    dot_file << "    rankdir=LR;\n";  // Left to right layout
    
    // Style definitions
    dot_file << "    node [shape=box, style=filled];\n";
    
    // Define nodes
    for (size_t i = 0; i < kernels.size(); i++) {
        dot_file << "    kernel_" << i << " [label=\"Kernel " << i << "\", fillcolor=lightblue];\n";
    }
    for (size_t i = 0; i < streams.size(); i++) {
        dot_file << "    stream_" << i << " [label=\"Stream " << i << "\", fillcolor=green];\n";
    }
    
    // Define edges
    for (const auto& conn : connections) {
        std::string src_name = conn.source.is_kernel() ? 
            "kernel_" + std::to_string(conn.source.index) :
            "stream_" + std::to_string(conn.source.index);
            
        std::string dst_name = conn.dest.is_kernel() ? 
            "kernel_" + std::to_string(conn.dest.index) :
            "stream_" + std::to_string(conn.dest.index);
        
        // Add port labels if they exist
        std::string label;
        if (!conn.source.port.empty() || !conn.dest.port.empty()) {
            label = " [label=\"" + 
                (conn.source.port.empty() ? "source" : conn.source.port) + " → " +
                (conn.dest.port.empty() ? "sink" : conn.dest.port) + "\"]";
        }
        
        dot_file << "    " << src_name << " -> " << dst_name << label << ";\n";
    }
    
    dot_file << "}\n";
    dot_file.close();
}

} // End namespace stream