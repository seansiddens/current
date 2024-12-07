#include "stream.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>

#include "impl/buffers/buffer.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "common/work_split.hpp"
#include "tt_metal/impl/device/device.hpp"

#include "common.hpp"

namespace current {


// Add a new input or output port to the kernel
void Kernel::add_input_port(const std::string& name, tt::DataFormat data_format)  {
    assert(input_ports.size() < MAX_INPUT_PORTS && "Kernel has too many input ports!");
    
    // Check if port name already exists in input ports
    for (const auto& port : input_ports) {
        assert(port.name != name && "Input port with this name already exists!");
    }
    // Check if port name already exists in output ports
    for (const auto& port : output_ports) {
        assert(port.name != name && "Port name already exists as an output port!");
    }
    
    input_ports.push_back({name, data_format});
}

void Kernel::add_output_port(const std::string& name, tt::DataFormat data_format) {
    assert(output_ports.size() < MAX_OUTPUT_PORTS && "Kernel has too many output ports!");
    
    // Check if port name already exists in output ports
    for (const auto& port : output_ports) {
        assert(port.name != name && "Output port with this name already exists!");
    }
    // Check if port name already exists in input ports
    for (const auto& port : input_ports) {
        assert(port.name != name && "Port name already exists as an input port!");
    }
    
    output_ports.push_back({name, data_format});
}

uint32_t Kernel::num_input_ports() const {
    return input_ports.size();
}

uint32_t Kernel::num_output_ports() const {
    return output_ports.size();
}

Map::Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams, uint32_t max_parallelization_factor) 
    : kernels(std::move(kernels)), streams(streams), max_parallelization_factor(max_parallelization_factor) {
    // Check that all streams have the same number of elements.
    for (size_t i = 1; i < streams.size(); i++) {
        // TODO: Eventually we want to support streams of different sizes e.g for reduction kernels,
        // but right now we just check that all streams have the same number of elements.
        assert(streams[i]->n_elements == streams[0]->n_elements && "All streams must have the same number of elements!");
    }
}

// TODO: Validate that port connections are valid (check that types are the same.)
void Map::add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in) {
    // TODO: Add error handling.
    auto src_kernel_idx = get_kernel_index(src);
    auto dst_kernel_idx = get_kernel_index(dst);
    Endpoint src_endpoint = {Endpoint::EndpointType::Kernel, src_kernel_idx, src_out};
    Endpoint dst_endpoint = {Endpoint::EndpointType::Kernel, dst_kernel_idx, dst_in};
    tt::log_info("[CURRENT] Adding connection from kernel {} to kernel {}", src_kernel_idx, dst_kernel_idx);
    add_connection(src_endpoint, dst_endpoint);
}

void Map::add_connection(Stream *src, Kernel *dst, std::string dst_in) {
    auto src_stream_idx = get_stream_index(src);
    auto dst_kernel_idx = get_kernel_index(dst);
    Endpoint src_endpoint = {Endpoint::EndpointType::Stream, src_stream_idx, ""};
    Endpoint dst_endpoint = {Endpoint::EndpointType::Kernel, dst_kernel_idx, dst_in};
    tt::log_info("[CURRENT] Adding connection from stream {} to kernel {}", src_stream_idx, dst_kernel_idx);
    add_connection(src_endpoint, dst_endpoint);
}

void Map::add_connection(Kernel *src, std::string src_out, Stream *dst) {
    auto src_kernel_idx = get_kernel_index(src);
    auto dst_stream_idx = get_stream_index(dst);
    Endpoint src_endpoint = {Endpoint::EndpointType::Kernel, src_kernel_idx, src_out};
    Endpoint dst_endpoint = {Endpoint::EndpointType::Stream, dst_stream_idx, ""};
    tt::log_info("[CURRENT] Adding connection from kernel {} to stream {}", src_kernel_idx, dst_stream_idx);
    // TODO: Need to do checks whether these are valid port names and that the ports have not already been connected.
    add_connection(src_endpoint, dst_endpoint);
}

// void Map::parallelize(std::vector<CoreCoord> &cores) {
//     auto total_cores = cores.size();

//     for (size_t i = 0; i < kernels.size(); i++) {

//     }
// }

void Map::execute() {
    check_connections();
    propagate_counts();

    // 1. Create device and program.
    auto device = tt_metal::CreateDevice(0);
    if (!device) {
        std::cerr << "Failed to create device!\n";
        exit(1);
    }
    auto program = tt_metal::CreateProgram();

    // 2. Core grid setup.
    // TODO: Have this configurable by user and dyanmic by runtime scheduling.
    // Write now just set to # of kernels we have.
    // runtime.num_cores = kernels.size();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_cores_x = compute_with_storage_grid_size.x;
    auto num_cores_y = compute_with_storage_grid_size.y;
    auto core_set = num_cores_to_corerange_set({0, 0}, kernels.size() * max_parallelization_factor, {num_cores_x, num_cores_y});

    runtime.emplace(Runtime {
        .device = device,
        .program = std::move(program),
        .num_cores_x = static_cast<uint32_t>(num_cores_x),
        .num_cores_y = static_cast<uint32_t>(num_cores_y),
        .core_set = std::move(core_set)
    });

    tt::log_info("[CURRENT] num_cores_x: {}, num_cores_y: {}", runtime->num_cores_x, runtime->num_cores_y);
    tt::log_info("[CURRENT] core_set: {}", runtime->core_set.str());
    tt::log_info("[CURRENT] Total cores: {}", runtime->core_set.num_cores());

    // 3. Input & Output DRAM buffer setup.
    for (size_t i = 0; i < streams.size(); i++) {
        auto stream = streams[i];
        tt_metal::InterleavedBufferConfig config = {
            .device = runtime->device,
            .size = stream->n_elements * stream->element_size,
            .page_size = stream->element_size * TILE_WIDTH * TILE_HEIGHT, // TODO: Not sure what is optimal for this.
            .buffer_type = tt_metal::BufferType::DRAM
        };
        std::cout << "STREAM " << i << ": size: " << config.size << std::endl;
        stream->device_buffer = tt_metal::CreateBuffer(config);
        // TODO: Does this need to be blocking?
        // TODO: What if there's a mismatch between the host data size and the device buffer size?
        tt_metal::EnqueueWriteBuffer(runtime->device->command_queue(), stream->device_buffer, stream->host_data, true);
        stream->device_buffer_address = stream->device_buffer->address();
        stream->device_buffer_noc_coordinates = stream->device_buffer->noc_coordinates();
    }

    // 4. Generate device kernels.
    generate_device_kernels();

    // Vector of cores we have availible to assign to kernels.
    std::vector<CoreCoord> cores = corerange_to_cores(runtime->core_set);

    // TODO: Bug with this when parallelizatino factor is not a multiple of num tiles??? Some sort of workload split bug.
    auto total_cores = cores.size();
    std::cout << "Total cores: " << total_cores << std::endl;
    size_t cores_used = 0;
    for (size_t i = 0; i < kernels.size(); i++) {
        // Each kernel gets mapped to a single core. 
        // We just assign to the next available core.
        // TODO: Look into core placement strategies (RaftLib thesis)
        // possibly doing automatic parallelization of kernels.
        // NOTE: This also requires that we need as many cores as kernels.
        size_t cores_availible = total_cores - cores_used;
        size_t cores_to_assign = std::min(cores_availible, (size_t)max_parallelization_factor);

        std::vector<CoreCoord> kernel_cores;
        for (size_t j = 0; j < cores_to_assign; j++) {
            kernel_cores.push_back(cores[cores_used + j]);
        }

        kernels[i]->core_spec = kernel_cores;
        cores_used += cores_to_assign;

        std::cout << "Kernel " << i << " assigned to cores: ";
        for (const auto& core : kernel_cores) {
            std::cout << "(" << core.x << ", " << core.y << ") ";
        }
        std::cout << std::endl;
    }

    // Print core distribution.

    for (size_t i = 0; i < kernels.size(); i++) {
        auto kernel = kernels[i];

        // Parallelize kernel across multiple cores.
        auto parallelization_factor = kernel->core_spec.size(); // # of cores this kernel is parallelized across.
        for (size_t j = 0; j < parallelization_factor; j++) {
            auto core = kernel->core_spec[j];

            // Create semaphores for each kernel.
            // TODO: Is overwriting across multiple cores, though we might not even need to store this ID for later.
            kernel->sender_semaphore_id = tt_metal::CreateSemaphore(runtime->program, core, INVALID);
            kernel->receiver_semaphore_id = tt_metal::CreateSemaphore(runtime->program, core, INVALID);
            kernel->l1_valid_value_semaphore_id = tt_metal::CreateSemaphore(runtime->program, core, VALID);

            auto incoming_connections = get_incoming_connections(kernel);
            auto outgoing_connections = get_outgoing_connections(kernel);

            // Create circular buffers for each incoming and outgoing connection.
            for (size_t k = 0; k < incoming_connections.size(); k++) {
                // Each port is typed with a specific data format. 
                auto cb_index = k + IN_CB_START;
                auto input_port = kernel->get_input_port(incoming_connections[k].dest.port);
                auto input_port_index = kernel->get_input_port_index(input_port.name);
                auto tile_size_bytes = TILE_WIDTH * TILE_HEIGHT * tt::datum_size(input_port.data_format);
                tt_metal::CircularBufferConfig cb_config = CircularBufferConfig(
                    TILES_PER_CB * tile_size_bytes,
                    {{cb_index, input_port.data_format}}
                ).set_page_size(cb_index, tile_size_bytes); // TODO: Not sure what to set this page size to.
                // TODO: Again, overwriting across multiple cores.
                kernel->input_ports[input_port_index].cb = tt_metal::CreateCircularBuffer(runtime->program, core, cb_config);

                // // Create mailbox buffer for incoming connections from another kernel.
                // if (incoming_connections[i].source.is_kernel()) {
                //     auto mailbox_config = tt_metal::InterleavedBufferConfig{
                //         .device = runtime.device,
                //         .size = tile_size_bytes,
                //         .page_size = tile_size_bytes,
                //         .buffer_type = tt_metal::BufferType::L1
                //     };
                //     kernel->input_ports[input_port_index].mailbox = tt_metal::CreateBuffer(mailbox_config);
                // }
            }

            for (size_t k = 0; k < outgoing_connections.size(); k++) {
                auto cb_index = k + OUT_CB_START;
                auto output_port = kernel->get_output_port(outgoing_connections[k].source.port);
                auto output_port_index = kernel->get_output_port_index(output_port.name);
                auto tile_size_bytes = TILE_WIDTH * TILE_HEIGHT * tt::datum_size(output_port.data_format);
                tt_metal::CircularBufferConfig cb_config = CircularBufferConfig(
                    TILES_PER_CB * tile_size_bytes,
                    {{cb_index, output_port.data_format}}
                ).set_page_size(cb_index, tile_size_bytes); // TODO: Not sure what to set this page size to.
                // TODO: Again, overwriting across multiple cores.
                kernel->output_ports[output_port_index].cb = tt_metal::CreateCircularBuffer(runtime->program, core, cb_config);
            }

            // Create device kernels.
            auto reader = tt_metal::CreateKernel(
                runtime->program,
                kernel->generated_reader_kernel_path,
                core,
                // TODO: Can also do compile-time args here? I think this might be useful.
                DataMovementConfig {
                    .processor = DataMovementProcessor::RISCV_0, 
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {},
                    .defines = {}
                } // TODO: What to do for this?
            );
            kernel->reader_kernel = reader;
            // DST is capable of storing 16 32x32 tiles of 2B datum size. In full mode, DST is not double buffered, so the full 16 tiles are available for LLKs to use. 
            // In half mode, we treat DST as a ping pong buffer, where each half contains 8 tiles. That means that an LLK can only index into 8 different tiles in DST. 
            // This is useful to overlap the MATH and PACK threads. 
            // Example: MATH will acquire the first half of DST and populate it with 8 tiles of output from some math LLK. 
            // MATH releases first half and acquires second half. 
            // PACK acquires first half and writes tiles from DST to L1. 
            // Meanwhile, MATH is producing results into the second half of DST. 
            // We continue ping ponging to keep MATH and PACK busy at the same time. 
            // I haven’t heard about “TILE” mode, not sure if that’s used anywhere. 
            // HALF mode is most common and afaik expected to be the default.
            auto compute = tt_metal::CreateKernel(
                runtime->program,
                kernel->generated_compute_kernel_path,
                core,
                ComputeConfig{
                    // TODO: Also need to figure out what the heck to do for this.
                    .math_fidelity = MathFidelity::LoFi,
                    .math_approx_mode = false,
                    .compile_args = {},
                    .defines = {}
                }
            );
            kernel->compute_kernel = compute;

            auto writer = tt_metal::CreateKernel(
                runtime->program,
                kernel->generated_writer_kernel_path,
                core,
                DataMovementConfig {
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = {},
                    .defines = {}
                }
            );
            kernel->writer_kernel = writer;

            // Set runtime args.
            std::vector<uint32_t> reader_args;
            std::vector<uint32_t> compute_args;
            for (const auto& connection : incoming_connections) {
                uint32_t n_tiles = connection.n_tiles / parallelization_factor; // TODO: Account for when n_tiles % parallelization_factor != 0.
                uint32_t tile_offset = n_tiles * j;
                std::cout << "n_tiles: " << n_tiles << " tile_offset: " << tile_offset << std::endl;
                if (connection.source.is_stream()) {
                    // For every incoming stream connection, we need to know how many tiles we expect to read and what the DRAM address is.
                    auto stream = streams[connection.source.index];
                    reader_args.push_back(n_tiles);
                    reader_args.push_back(stream->device_buffer_address);
                    reader_args.push_back(tile_offset);
                    compute_args.push_back(n_tiles); // Compute also needs to know how many tiles to read in.
                } else {
                    // Incoming connection is another kernel. 
                    // TODO: I'm not sure if this is correct with the way I'm doing parallelization. Would make more sense to transform the program graph itself.
                    auto sender = kernels[connection.source.index];
                    CoreCoord sender_core = sender->core_spec[j];
                    uint32_t sender_x = (uint32_t)runtime->device->worker_core_from_logical_core(sender_core).x;
                    uint32_t sender_y = (uint32_t)runtime->device->worker_core_from_logical_core(sender_core).y;
                    reader_args.push_back(n_tiles);
                    reader_args.push_back(sender_x);
                    reader_args.push_back(sender_y);
                    reader_args.push_back(kernel->sender_semaphore_id);
                    reader_args.push_back(kernel->receiver_semaphore_id);
                    reader_args.push_back(kernel->l1_valid_value_semaphore_id);
                    compute_args.push_back(n_tiles);
                }
            }
            SetRuntimeArgs(runtime->program, kernel->reader_kernel, core, reader_args);
            SetRuntimeArgs(runtime->program, kernel->compute_kernel, core, compute_args);

            std::vector<uint32_t> writer_args;
            for (const auto& connection : outgoing_connections) {
                uint32_t n_tiles = connection.n_tiles / parallelization_factor; // TODO: Account for when n_tiles % parallelization_factor != 0.
                uint32_t tile_offset = n_tiles * j; // TODO: Only works if total work is evenly divisible by parallelization factor.
                if (connection.dest.is_stream()) {
                    // TODO: Here we are explicitly setting the # of tiles we expect to write to be the same as the capacity of the stream.
                    // This can get a bit tricky if the compute does any sort of reduction and the user does not correctly set the capacity.
                    // I think if reduction kernels get implemented then we need a way of automatically determining the # of tiles to write at each stage of the program.
                    auto stream = streams[connection.dest.index];
                    writer_args.push_back(n_tiles);
                    writer_args.push_back(stream->device_buffer_address);
                    writer_args.push_back(tile_offset);
                } else {
                    // Outgoing connection is another kernel.
                    auto receiver = kernels[connection.dest.index];
                    CoreCoord receiver_core = receiver->core_spec[j];
                    uint32_t receiver_x = (uint32_t)runtime->device->worker_core_from_logical_core(receiver_core).x;
                    uint32_t receiver_y = (uint32_t)runtime->device->worker_core_from_logical_core(receiver_core).y;
                    uint32_t receiver_mailbox_addr = kernel->get_input_port(connection.dest.port).mailbox->address();
                    writer_args.push_back(n_tiles);
                    writer_args.push_back(receiver_x);
                    writer_args.push_back(receiver_y);
                    // writer_args.push_back(receiver_mailbox_addr);
                    writer_args.push_back(kernel->sender_semaphore_id);
                    writer_args.push_back(kernel->receiver_semaphore_id);
                    writer_args.push_back(kernel->l1_valid_value_semaphore_id);
                }
            }
            SetRuntimeArgs(runtime->program, kernel->writer_kernel, core, writer_args);
        }
    }

    // Collect benchmark metrics.
    auto start = std::chrono::high_resolution_clock::now();
    tt_metal::EnqueueProgram(runtime->device->command_queue(), runtime->program, false);
    tt_metal::Finish(runtime->device->command_queue());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    tt::log_info("[CURRENT] Program execution completed! Time taken: {} milliseconds", duration.count());
    auto total_bytes = streams[0]->n_elements * streams[0]->element_size;
    tt::log_info("[CURRENT] Total bytes transferred: {}", total_bytes);
    double total_seconds = duration.count() / 1000.0;
    tt::log_info("[CURRENT] Total throughput: {} GB/s", (total_bytes / total_seconds) / 1e9);
    // tt_metal::CloseDevice(runtime.device);
}

Map::~Map() {
    if (runtime) {
        tt_metal::CloseDevice(runtime->device);
    }
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
        if (connection.dest.index == kernel_idx && connection.dest.endpoint_type == Endpoint::EndpointType::Kernel) {
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
        if (connection.source.index == kernel_idx && connection.source.endpoint_type == Endpoint::EndpointType::Kernel) {
            outgoing_connections.push_back(connection);
        }
    }
    return outgoing_connections;
}

std::string data_format_to_string(tt::DataFormat data_format) {
    // std::cout << "Data format: " << data_format << "\n";
    switch (data_format) {
        case tt::DataFormat::Float16_b: return "DataFormat::Float16_b";
        case tt::DataFormat::Bfp8_b: return "DataFormat::Bfp8_b";
        default:
            std::cerr << "Unsupported data format!\n";
            exit(1);
    }
}

void Map::generate_reader_device_kernel(
    Kernel *kernel,
    std::vector<Connection> incoming_connections
) {
    // Generate reader kernel.
    std::stringstream rs;
    rs << "#include <cstdint>\n";
    rs << "#include \"dataflow_api.h\"\n";
    rs << "#include \"debug/dprint.h\"\n";
    rs << "#include \"hostdevcommon/common_values.hpp\"\n";
    rs << "#include \"debug/dprint.h\"\n";
    rs << "void kernel_main() {\n";

    // Reader params from kernel args
    uint32_t total_args = 0;

    for (size_t i = 0; i < incoming_connections.size(); i++) {
        auto connection = incoming_connections[i];
        if (connection.source.is_stream()) {
            auto stream = streams[connection.source.index];
            // Total # of tiles this kernel will read from this stream.
            // Stream -> Kernel, get the input port index.
            auto port = kernel->get_input_port(connection.dest.port);
            rs << "    uint32_t " << port.name << "_ntiles = get_arg_val<uint32_t>(" << total_args << ");\n";
            rs << "    DPRINT << \"READER0: " << port.name << "_ntiles: \" << " << port.name << "_ntiles << ENDL();\n";
            total_args++;
            // For every incoming stream connection, we need to get it's address and create an address generator.
            rs << "    uint32_t " << port.name << "_addr = get_arg_val<uint32_t>(" << total_args << ");\n";
            rs << "    DPRINT << \"READER0: " << port.name << "_addr: \" << " << port.name << "_addr << ENDL();\n";
            total_args++;
            rs << "    uint32_t " << port.name << "_tile_offset = get_arg_val<uint32_t>(" << total_args << ");\n";
            rs << "    DPRINT << \"READER0: " << port.name << "_tile_offset: \" << " << port.name << "_tile_offset << ENDL();\n";
            total_args++;
            // Address generator.
            // TODO: Do we need this? How does this even work?
            rs << "    const InterleavedAddrGenFast<true> " << port.name << "_addr_gen = {\n";
            rs << "        .bank_base_address = " << port.name << "_addr, \n";
            rs << "        .page_size = " << TILE_WIDTH * TILE_HEIGHT * stream->element_size << ", \n";
            rs << "        .data_format = " << data_format_to_string(stream->data_format) << ", \n";
            rs << "    };\n\n";
        } else {
            auto port = kernel->get_input_port(connection.dest.port);
            rs << "    uint32_t " << port.name << "_ntiles = get_arg_val<uint32_t>(" << total_args << ");\n";
            rs << "    DPRINT << \"READER1: " << port.name << "_ntiles: \" << " << port.name << "_ntiles << ENDL();\n";
            total_args++;
            rs << "    uint32_t " << port.name << "_sender_noc_x = get_arg_val<uint32_t>(" << total_args << ");\n";
            rs << "    DPRINT << \"READER1: " << port.name << "_sender_noc_x: \" << " << port.name << "_sender_noc_x << ENDL();\n";
            total_args++;
            rs << "    uint32_t " << port.name << "_sender_noc_y = get_arg_val<uint32_t>(" << total_args << ");\n";
            rs << "    DPRINT << \"READER1: " << port.name << "_sender_noc_y: \" << " << port.name << "_sender_noc_y << ENDL();\n";
            total_args++;
            rs << "    uint32_t " << port.name << "_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(" << total_args << "));\n";
            rs << "    DPRINT << \"READER1: " << port.name << "_sender_semaphore_addr: \" << " << port.name << "_sender_semaphore_addr << ENDL();\n";
            total_args++;
            rs << "    uint32_t " << port.name << "_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(" << total_args << "));\n";
            rs << "    DPRINT << \"READER1: " << port.name << "_receiver_semaphore_addr: \" << " << port.name << "_receiver_semaphore_addr << ENDL();\n";
            total_args++;
            // rs << "    uint32_t " << port.name << "_l1_valid_value_semaphore_id = get_arg_val<uint32_t>(" << total_args << ");\n";
            // total_args++;

            rs << "    volatile tt_l1_ptr uint32_t* " << port.name << "_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(" << port.name << "_receiver_semaphore_addr);\n";
            rs << "    uint64_t " << port.name << "_sender_semaphore_noc_addr = get_noc_addr(" << port.name << "_sender_noc_x, " << port.name << "_sender_noc_y, " << port.name << "_sender_semaphore_addr);\n";
            rs << "    DPRINT << \"READER1: " << port.name << "_sender_semaphore_noc_addr: \" << " << port.name << "_sender_semaphore_noc_addr << ENDL();\n";
            rs << "\n";
        }
    }

    // Circular buffers.
    uint32_t num_input_cbs = 0;
    for (size_t i = 0; i < incoming_connections.size(); i++) {
        // Assign CBs to input ports in iteration order.
        auto port = kernel->get_input_port(incoming_connections[i].dest.port);
        rs << "    constexpr uint32_t " << port.name << " = " << num_input_cbs << ";\n";
        rs << "    DPRINT << \"READER0: " << port.name << " tile_size: \" << get_tile_size(" << port.name << ") << ENDL();\n";
        num_input_cbs++;
    }
    rs << "\n";

    if (incoming_connections.size() > 0) {
        // Input tile stream loop.
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            // Generate a counter variable for every incoming connection. 
            // Each incoming connection maps to a specific input port, which is managed by a CB.
            auto connection = incoming_connections[i];
            auto port = kernel->get_input_port(connection.dest.port);
            rs << "    uint32_t " << port.name << "_count = 0;\n";
        }

        // The break condition is when we've read the expected # of tiles from each input port.
        // TODO: The only case when the incoming stream sizes would be different is if we are doing some sort of reduction on a stream 
        // e.g for each record of stream A we are dequining 4 records from stream B.
        // In this case, we might be requesting/dequing N tiles at once. This needs to somehow be handled in the compute kernel.
        // Right now even though we are acting as if the streams are different sizes, the compute kernel is still acting as if they are the same size.
        std::string break_condition = "";
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            auto port = kernel->get_input_port(incoming_connections[i].dest.port);
            break_condition += port.name + "_count < " + port.name + "_ntiles";
            if (i != incoming_connections.size() - 1) {
                break_condition += " && ";
            }
        }

        // rs << "    for(uint32_t i = 0; i < source0_n_tiles; i++) {\n";
        rs << "    while(" << break_condition << ") {\n";

        // Wait for space in CBs
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            auto connection = incoming_connections[i];
            if (!connection.source.is_stream()) {
                continue;
            }
            auto port = kernel->get_input_port(incoming_connections[i].dest.port);
            rs << "        if (" << port.name << "_count < " << port.name << "_ntiles) {\n";
            rs << "            cb_reserve_back(" << port.name << ", 1);\n";
            rs << "        }\n";
        }
        // Read tile into CB from DRAM.
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            auto connection = incoming_connections[i];
            if (!connection.source.is_stream()) {
                continue;
            }
            auto port = kernel->get_input_port(incoming_connections[i].dest.port);
            rs << "        if (" << port.name << "_count < " << port.name << "_ntiles) {\n";
            rs << "            uint32_t " << port.name << "_write_ptr = get_write_ptr(" << port.name << ");\n";
            rs << "            uint32_t id = " << port.name << "_tile_offset + " << port.name << "_count;\n";
            rs << "            DPRINT << \"READER0: id: \" << id << ENDL();\n";
            rs << "            noc_async_read_tile(id, " <<  port.name << "_addr_gen, " << port.name << "_write_ptr);\n";
            rs << "        }\n";
        }

        // Wait until tile reads are done.
        // TODO: Don't do if not reading from stream.
        rs << "\n";
        rs << "        noc_async_read_barrier();\n";
        rs << "\n";

        // Push tiles into CBs and increment counters.
        // Signals to compute engine that a tile is ready to be processed.
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            auto connection = incoming_connections[i];
            if (!connection.source.is_stream()) {
                continue;
            }
            auto port = kernel->get_input_port(incoming_connections[i].dest.port);
            rs << "        if (" << port.name << "_count < " << port.name << "_ntiles) {\n";
            rs << "            cb_push_back(" << port.name << ", 1);\n";
            rs << "            " << port.name << "_count++;\n";
            rs << "        }\n";
        }

        // Do receiver stuff.
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            auto connection = incoming_connections[i];
            if (connection.source.is_stream()) {
                continue;
            }
            auto port = kernel->get_input_port(incoming_connections[i].dest.port);
            // Wait for space in CB.
            rs << "        DPRINT << \"READER1: Waiting for space in CB " << port.name << "\" << ENDL();\n";
            rs << "        cb_reserve_back(" << port.name << ", 1);\n";
            // Reset receiver's own semaphore value to INVALID.
            rs << "        noc_semaphore_set(" << port.name << "_receiver_semaphore_addr_ptr, INVALID);\n";
            // Tell sender we're ready -- atomic increment sender's semaphore.
            rs << "        DPRINT << \"READER1: Telling sender we're ready\" << ENDL();\n";
            rs << "        noc_semaphore_inc(" << port.name << "_sender_semaphore_noc_addr, 1);\n";
            // Wait on receiver's own semaphore value to become VALID (set by sender after it sends the data).
            rs << "        DPRINT << \"READER1: Waiting on receiver's semaphore\" << ENDL();\n";
            rs << "        noc_semaphore_wait(" << port.name << "_receiver_semaphore_addr_ptr, VALID);\n";
            rs << "        DPRINT << \"READER1: Receiver's semaphore is VALID!\" << ENDL();\n";
            // Push tile into CB.
            rs << "        DPRINT << \"READER1: Pushing tile into CB\" << ENDL();\n";
            rs << "        cb_push_back(" << port.name << ", 1);\n";
            // Increment counter.
            rs << "        " << port.name << "_count++;\n";
        }



        rs << "    }\n";
        // End tile stream loop.
    }

    rs << "}\n";
    rs << "\n";

    std::string filename = "reader" + std::to_string(get_kernel_index(kernel)) + ".cpp";
    kernel->generated_reader_kernel_path = GENERATED_KERNELS_PATH / filename;
    auto reader_kernel_file = std::ofstream(kernel->generated_reader_kernel_path);
    if (!reader_kernel_file.is_open()) {
        tt::log_error("[CURRENT] Failed to open file for writing: {}", kernel->generated_reader_kernel_path);
        exit(1);
    }
    reader_kernel_file << rs.str();
    reader_kernel_file.close();
}

void Map::generate_compute_device_kernel(
    Kernel *kernel,
    std::vector<Connection> incoming_connections,
    std::vector<Connection> outgoing_connections
) {
    std::stringstream cs; 
    // Includes
    cs << "#include \"compute_kernel_api/common.h\"\n";
    cs << "#include \"compute_kernel_api/tile_move_copy.h\"\n";
    cs << "#include \"compute_kernel_api/eltwise_binary.h\"\n";
    cs << "#include \"compute_kernel_api/eltwise_unary/eltwise_unary.h\"\n";
    cs << "#include \"compute_kernel_api/matmul.h\"\n";
    cs << "#include \"compute_kernel_api.h\"\n";
    cs << "#include \"cmath_common.h\"\n";
    cs << "#include \"sfpi.h\"\n";
    cs << "#include \"debug/dprint.h\"\n";
    cs << "\n";

    // SFPU computation
    cs << "namespace sfpi {\n";
    // cs << "template< int ITERATIONS = 16 >\n";
    cs << "sfpi_inline void compute() {\n";
    // Set destination write address.
    cs << "    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);\n";
    // If we don't have a specifed compute kernel, don't generate anything.
    if (!kernel->sfpi_kernel_string.empty()) {
        // TODO: Do a better optimization if we don't have a compute kernel.
        // Can probably avoid any call to the sfpi function, don't need to do sfpi init? idk
        cs << "    for (int i = 0; i < 32; i++) {\n";
        // Get input variables.
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            cs << "        vFloat in" << i << " = dst_reg[" << i << " * 32 + i];\n";
        }
        // Declare output variables.
        for (size_t i = 0; i < outgoing_connections.size(); i++) {
            cs << "        vFloat out" << i << ";\n";
        }
        cs << kernel->sfpi_kernel_string;
        // Assign output variables.
        for (size_t i = 0; i < outgoing_connections.size(); i++) {
            cs << "        dst_reg[" << i << " * 32 + i] = out" << i << ";\n";
        }
        cs << "    }\n";

    }
    // cs << "    for (int i = 0; i < ITERATIONS; i++) {\n";
    // cs << "        vFloat in = dst_reg[i];\n";
    // cs << "        vFloat a = in + 1.0f;\n";
    // cs << "        vFloat out = a;\n";
    // cs << "        dst_reg[i] = out;\n";
    // cs << "    }\n";
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
    for (size_t i = 0; i < incoming_connections.size(); i++) {
        // Assign CBs to input ports in iteration order.
        auto port = kernel->get_input_port(incoming_connections[i].dest.port);
        cs << "    constexpr uint32_t " << port.name << " = " << num_input_cbs << ";\n";
        num_input_cbs++;
    }
    cs << "\n";
    uint32_t num_output_cbs = OUT_CB_START; // Output CBs start at index 16.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        // Assign CBs to output ports in iteration order.
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        cs << "    constexpr uint32_t " << port.name << " = " << num_output_cbs << ";\n";
        num_output_cbs++;
    }
    cs << "\n";



    // Initialize SFPU.
    for (size_t i = 0; i < incoming_connections.size(); i++) {
        auto port = kernel->get_input_port(incoming_connections[i].dest.port);
        cs << "    init_sfpu(" << port.name << ");\n";
    }
    if (kernel->do_matmul) {
        // TODO: This assumes that we are matmuling port 0 and 1.
        cs << "    mm_init();\n";
    }
    cs << "\n";

    // cs << "    copy_tile_init();\n";

    // Tile stream loop
    // TODO: Right now just going to assume that all streams have the same number of tiles.
    cs << "    for(uint32_t i = 0; i < n_tiles; i++) {\n";
    // Wait for tiles to be read in CBs.
    for (size_t i = 0; i < incoming_connections.size(); i++) {
        auto port = kernel->get_input_port(incoming_connections[i].dest.port);
        cs << "        cb_wait_front(" << port.name << ", 1);\n";
    }
    cs << "\n";

    // Copy tiles from CBs to SFPU registers.
    cs << "        tile_regs_acquire();\n";
    if (kernel->do_matmul) {
        // TODO: Again, assuming that we are matmuling port 0 and 1 and we don't have any other input ports.
        cs << "        ckernel::matmul_tiles(" << kernel->get_input_port(incoming_connections[0].dest.port).name << ", " << kernel->get_input_port(incoming_connections[1].dest.port).name << ", 0, 0, 0, 0);\n";
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            cs << "        cb_pop_front(" << kernel->get_input_port(incoming_connections[i].dest.port).name << ", 1);\n";
        }
    } else {
        for (size_t i = 0; i < incoming_connections.size(); i++) {
            // auto port = kernel->get_input_port(incoming_connections[incoming_connections.size() - i - 1].dest.port);
            auto port = kernel->get_input_port(incoming_connections[i].dest.port);
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
            // cs << "        copy_tile(" << port.name << ", 0, " << incoming_connections.size() - i - 1 << ");\n";
            cs << "        copy_tile(" << port.name << ", 0, " << i << ");\n";
            // cs << "        UNPACK(( llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(" << port.name << ", 0)  ));\n";
            // TODO: Not sure how this works in the context of the compute cores. Which core is doing this pop?
            cs << "        cb_pop_front(" << port.name << ", 1);\n";
        }
    }

    // for (size_t i = 0; i < incoming_connections.size(); i++) {
    //     auto port = kernel->get_input_port(incoming_connections[i].dest.port);
    //     cs << "        MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, DST_ACCUM_MODE, UnpackToDestEn>(" << i << ", " << port.name << ") ));\n";
    // }


    cs << "\n";
    // cs << "        tile_regs_acquire();\n";
    cs << "        MATH((sfpi::compute()));\n";
    cs << "        tile_regs_commit();\n";
    cs << "\n";

    // // Computation finished, pop tiles from input CBs
    // // TODO: This may be able to be re-ordered?
    // for (size_t i = 0; i < incoming_connections.size(); i++) {
    //     auto port = kernel->get_input_port(incoming_connections[i].dest.port);
    //     cs << "        cb_pop_front(" << port.name << ", 1);\n";
    // }
    // cs << "\n";

    // Packer waits here until the SFPU is done.
    cs << "        tile_regs_wait();\n";
    // Reserve space in output CBs.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        cs << "        cb_reserve_back(" << port.name << ", 1);\n";
    }
    cs << "\n";
    // Pack tiles into output CBs.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        cs << "        pack_tile(" << i << ", " << port.name << ");\n";
    }
    cs << "\n";
    // Announce that the output tiles are ready.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        cs << "        cb_push_back(" << port.name << ", 1);\n";
    }
    cs << "\n";

    // Packer releases the SFPU registers.
    cs << "        tile_regs_release();\n";

    // End tile stream loop.
    cs << "    }\n";
    cs << "    DPRINT << \"COMPUTE0: Done\" << ENDL();\n";

    // End main.
    cs << "}\n";
    cs << "}\n";
    cs << "\n";

    // Save to file.
    std::string filename = "compute" + std::to_string(get_kernel_index(kernel)) + ".cpp";
    kernel->generated_compute_kernel_path = GENERATED_KERNELS_PATH / filename;
    auto compute_kernel_file = std::ofstream(kernel->generated_compute_kernel_path);
    if (!compute_kernel_file.is_open()) {
        tt::log_error("[CURRENT] Failed to open file for writing: {}", kernel->generated_compute_kernel_path);
        exit(1);
    }
    compute_kernel_file << cs.str();
    compute_kernel_file.close();
}

void Map::generate_writer_device_kernel(
    Kernel *kernel,
    std::vector<Connection> outgoing_connections
) {

    std::stringstream ws;
    // Includes.
    ws << "#include <cstdint>\n";
    ws << "#include \"hostdevcommon/common_values.hpp\"\n";
    ws << "#include \"dataflow_api.h\"\n";
    ws << "#include \"debug/dprint.h\"\n";
    ws << "\n";

    // Main 
    ws << "void kernel_main() {\n";

    // Writer params from kernel args
    uint32_t total_args = 0;
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (connection.dest.is_stream()) {
            // TODO: Here we are using the capacity of the stream we are writing to in order to determine how many tiles we need to write.
            // The issue with this is that if compute does any sort of reduction, then the capacity of the stream will be incorrect (unless it's explicitly set to match).
            // Need to think about how to do this automatically (e.g analyzing the # of tiles we stream in and out for each tile).
            auto stream = streams[connection.dest.index];
            // Kernel -> Stream, get the output port index.
            auto port = kernel->get_output_port(connection.source.port);
            // Total # of tiles this kernel will write to this stream.
            ws << "    uint32_t " << port.name << "_ntiles = get_arg_val<uint32_t>(" << total_args << ");\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_ntiles: \" << " << port.name << "_ntiles << ENDL();\n";
            total_args++;
            // For every outgoing stream connection, we need to get it's address and create an address generator.
            ws << "    uint32_t " << port.name << "_addr = get_arg_val<uint32_t>(" << total_args << ");\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_addr: \" << " << port.name << "_addr << ENDL();\n";
            total_args++;
            ws << "    uint32_t " << port.name << "_tile_offset = get_arg_val<uint32_t>(" << total_args << ");\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_tile_offset: \" << " << port.name << "_tile_offset << ENDL();\n";
            total_args++;
            // Address generator.
            // TODO: Do we need this? How does this even work?
            ws << "    const InterleavedAddrGenFast<true> " << port.name << "_addr_gen = {\n";
            ws << "        .bank_base_address = " << port.name << "_addr, \n";
            ws << "        .page_size = " << TILE_WIDTH * TILE_HEIGHT * stream->element_size << ", \n";
            ws << "        .data_format = " << data_format_to_string(stream->data_format) << ", \n";
            ws << "    };\n\n";
        } else {
            // TODO: Handle outgoing kernel connections.
            auto port = kernel->get_output_port(connection.source.port);
            ws << "    uint32_t " << port.name << "_ntiles = get_arg_val<uint32_t>(" << total_args << ");\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_ntiles: \" << " << port.name << "_ntiles << ENDL();\n";
            total_args++;
            ws << "    uint32_t " << port.name << "_receiver_noc_x = get_arg_val<uint32_t>(" << total_args << ");\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_receiver_noc_x: \" << " << port.name << "_receiver_noc_x << ENDL();\n";
            total_args++;
            ws << "    uint32_t " << port.name << "_receiver_noc_y = get_arg_val<uint32_t>(" << total_args << ");\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_receiver_noc_y: \" << " << port.name << "_receiver_noc_y << ENDL();\n";
            total_args++;
            ws << "    uint32_t " << port.name << "_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(" << total_args << "));\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_sender_semaphore_addr: \" << " << port.name << "_sender_semaphore_addr << ENDL();\n";
            total_args++;
            ws << "    uint32_t " << port.name << "_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(" << total_args << "));\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_receiver_semaphore_addr: \" << " << port.name << "_receiver_semaphore_addr << ENDL();\n";
            total_args++;
            ws << "    uint32_t " << port.name << "_l1_valid_value_addr = get_semaphore(get_arg_val<uint32_t>(" << total_args << "));\n";
            ws << "    DPRINT << \"WRITER0: " << port.name << "_l1_valid_value_addr: \" << " << port.name << "_l1_valid_value_addr << ENDL();\n";
            total_args++;
            ws << "\n";

            // Initialized to zero by host before program launch.
            ws << "    volatile tt_l1_ptr uint32_t* " << port.name << "_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(" << port.name << "_sender_semaphore_addr);\n";
            // Local valid value in L1.
            ws << "    volatile tt_l1_ptr uint32_t* " << port.name << "_l1_valid_value_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(" << port.name << "_l1_valid_value_addr);\n";
            ws << "    *(" << port.name << "_l1_valid_value_addr_ptr) = VALID;\n";
            ws << "\n";
        }
    }

    // Circular buffers.
    uint32_t num_output_cbs = OUT_CB_START; // Output CBs start at index 16.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        // Assign CBs to input ports in iteration order.
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        ws << "    constexpr uint32_t " << port.name << " = " << num_output_cbs << ";\n";
        ws << "    DPRINT << \"WRITER0: " << port.name << " tile_size: \" << get_tile_size(" << port.name << ") << ENDL();\n";
        num_output_cbs++;
    }
    ws << "\n";

    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (connection.dest.is_stream()) {
            continue;
        }
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        ws << "    uint64_t " << port.name << "_receiver_semaphore_noc_addr = get_noc_addr(" << port.name << "_receiver_noc_x, " << port.name << "_receiver_noc_y, " << port.name << "_receiver_semaphore_addr);\n";
        ws << "    DPRINT << \"WRITER0: " << port.name << "_receiver_semaphore_noc_addr: \" << " << port.name << "_receiver_semaphore_noc_addr << ENDL();\n";
    }
    ws << "\n";
    // Output tile stream loop.
    // TODO: Handle multiple output ports with DIFFERENT n_tiles.
    // In the loop, need to keep track of how many tiles we've written to each output port.
    // Break condition is when we've written the expected # of tiles to each output port.
    ws << "    for(uint32_t i = 0; i < " << outgoing_connections[0].source.port << "_ntiles; i++) {\n";
    // Wait tiles to arrive in CBs
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (!connection.dest.is_stream()) {
            continue;
        }
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        ws << "        DPRINT << \"WRITER0: Waiting for tile in CB " << port.name << "\" << ENDL();\n";
        ws << "        cb_wait_front(" << port.name << ", 1);\n";
    }

    // Write tiles to DRAM.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (!connection.dest.is_stream()) {
            continue;
        }
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        ws << "        uint32_t " << port.name << "_read_ptr = get_read_ptr(" << port.name << ");\n";
        ws << "        uint32_t id = " << port.name << "_tile_offset + i;\n";
        ws << "        DPRINT << \"WRITER0: id: \" << id << ENDL();\n";
        ws << "        noc_async_write_tile(id, " << port.name << "_addr_gen, " << port.name << "_read_ptr);\n";
    }
    // Wait until tile writes are done.
    ws << "\n";
    ws << "        noc_async_write_barrier();\n";
    ws << "\n";
    // // TODO: Potentially slower than just using noc_async_write_flushed().
    // // Might not even have to use until the last tile is written.

    // Mark the tiles as consumed.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (!connection.dest.is_stream()) {
            continue;
        }
        auto port = kernel->get_output_port(outgoing_connections[i].source.port);
        ws << "        cb_pop_front(" << port.name << ", 1);\n";
    }

    // Handle producer-consumer pattern.
    for (size_t i = 0; i < outgoing_connections.size(); i++) {
        auto connection = outgoing_connections[i];
        if (connection.dest.is_stream()) {
            continue;
        }
        // This kernel's port that we are sending out of.
        auto sender_port = kernel->get_output_port(outgoing_connections[i].source.port);

        // The input port of the kernel we are sending to.
        auto receiver_port = kernel->get_input_port(outgoing_connections[i].dest.port);
        // Wait until receiver has set the sender's semaphore to 1, which means receiver has reserved space in their CB.
        ws << "        DPRINT << \"WRITER0: Waiting for sender's semaphore to be set to 1\" << ENDL();\n";
        ws << "        noc_semaphore_wait(" << sender_port.name << "_sender_semaphore_addr_ptr, 1);\n";

        // Wait for data to be ready to be sent.
        ws << "        DPRINT << \"WRITER0: Waiting for data to be ready to be sent from CB\" << ENDL();\n";
        ws << "        cb_wait_front(" << sender_port.name << ", 1);\n";
        ws << "        uint32_t " << sender_port.name << "_read_ptr = get_read_ptr(" << sender_port.name << ");\n";

        // We have the data ready to be sent (at l1_addr), we can send it to the receiver.
        // We need to get a read_ptr to the CB that will be used by the receiver.
        // TODO: Need to figure out how to do this. RN I'm just going to assume the receiver is using in0 CB.
        // This might assume that the CBs are set up in the same exact way on both tiles?
        ws << "        uint32_t " << sender_port.name << "_receiver_read_ptr = get_read_ptr(0);\n";
        ws << "        uint64_t " << sender_port.name << "_receiver_noc_addr = get_noc_addr(" << sender_port.name << "_receiver_noc_x, " << sender_port.name << "_receiver_noc_y, " << sender_port.name << "_receiver_read_ptr);\n";
        ws << "        DPRINT << \"WRITER0: " << sender_port.name << "_receiver_noc_addr: \" << " << sender_port.name << "_receiver_noc_addr << ENDL();\n";
        // TODO: Don't hardcode the tile size, use elment size of stream data.
        ws << "        noc_async_write(" << sender_port.name << "_read_ptr, " << sender_port.name << "_receiver_noc_addr, " << TILE_WIDTH * TILE_HEIGHT * 2 << ");\n";
        ws << "        DPRINT << \"WRITER0: Writing to receiver's CB\" << ENDL();\n";

        // Set the sender's semaphore back to 0 for the next block.
        ws << "        noc_semaphore_set(" << sender_port.name << "_sender_semaphore_addr_ptr, 0);\n";
        ws << "        DPRINT << \"WRITER0: Set sender's semaphore back to 0\" << ENDL();\n";

        // Set the receiver's semaphore so that it knows that data has been written to the CB
        // must use noc_semaphore_set_remote and not noc_semaphore_inc in the sender
        // because we need to ensure that data is written to the remote CB before we set the semaphore
        // noc_async_write and noc_semaphore_set_remote are ordered
        ws << "        noc_semaphore_set_remote(" << sender_port.name << "_l1_valid_value_addr, " << sender_port.name << "_receiver_semaphore_noc_addr);\n";
        ws << "        DPRINT << \"WRITER0: Set receiver's semaphore to VALID\" << ENDL();\n";

        ws << "        cb_pop_front(" << sender_port.name << ", 1);\n";
        ws << "        DPRINT << \"WRITER0: Popped front of CB " << sender_port.name << "\" << ENDL();\n";
    }

    ws << "    }\n";
    ws << "    DPRINT << \"WRITER0: Done\" << ENDL();\n";
    // End tile stream loop.

    //End Main
    ws << "}\n";
    ws << "\n";

    // Save to file.
    std::string filename = "writer" + std::to_string(get_kernel_index(kernel)) + ".cpp";
    kernel->generated_writer_kernel_path = GENERATED_KERNELS_PATH / filename;
    auto writer_kernel_file = std::ofstream(kernel->generated_writer_kernel_path);
    if (!writer_kernel_file.is_open()) {
        tt::log_error("[CURRENT] Failed to open file for writing: {}", kernel->generated_writer_kernel_path);
        exit(1);
    }
    writer_kernel_file << ws.str();
    writer_kernel_file.close();
}

void Map::generate_device_kernels() {
    for (int i = 0; i < kernels.size(); i++) {
        Kernel *kernel = kernels[i];
        auto incoming_connections = get_incoming_connections(kernel);
        auto outgoing_connections = get_outgoing_connections(kernel);

        generate_reader_device_kernel(kernel, incoming_connections);
        generate_compute_device_kernel(kernel, incoming_connections, outgoing_connections);
        generate_writer_device_kernel(kernel, outgoing_connections);
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

void Map::check_connections() {
    bool flag = false;

    // Check all kernel ports have connections
    for (size_t kernel_idx = 0; kernel_idx < kernels.size(); kernel_idx++) {
        Kernel* kernel = kernels[kernel_idx];
        
        // Check input ports
        for (const auto& input_port : kernel->input_ports) {
            bool found = false;
            for (const auto& conn : connections) {
                if (conn.dest.is_kernel() && 
                    conn.dest.index == kernel_idx && 
                    conn.dest.port == input_port.name) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                tt::log_warning("[CURRENT] Kernel {} input port '{}' has no connection", kernel_idx, input_port.name);
            }
        }

        // Check output ports
        for (const auto& output_port : kernel->output_ports) {
            bool found = false;
            for (const auto& conn : connections) {
                if (conn.source.is_kernel() && 
                    conn.source.index == kernel_idx && 
                    conn.source.port == output_port.name) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                tt::log_warning("[CURRENT] Kernel {} output port '{}' has no connection", kernel_idx, output_port.name);
                flag = true;
            }
        }
    }

    // Check all streams have connections
    for (size_t stream_idx = 0; stream_idx < streams.size(); stream_idx++) {
        bool found = false;
        for (const auto& conn : connections) {
            if ((conn.source.is_stream() && conn.source.index == stream_idx) ||
                (conn.dest.is_stream() && conn.dest.index == stream_idx)) {
                found = true;
                break;
            }
        }
        if (!found) {
            tt::log_warning("[CURRENT] Stream {} has no connections", stream_idx);
            flag = true;
        }
    }

    assert(!flag && "Missing connections in map!");
}

std::vector<uint32_t> Map::read_stream(Stream *stream) {
    auto it = std::find(streams.begin(), streams.end(), stream);
    assert(it != streams.end() && "Stream not found in map!");

     // Verify this stream is actually a destination in our connections
    bool is_output = false;
    for (const auto& conn : connections) {
        if (conn.dest.is_stream() && streams[conn.dest.index] == stream) {
            is_output = true;
            break;
        }
    }
    assert(is_output && "Cannot get output from a stream that isn't a destination!");

    std::vector<uint32_t> out;
    tt_metal::EnqueueReadBuffer(runtime->device->command_queue(), stream->device_buffer, out, true);
    return out;
}

void Map::propagate_counts() {
    // TODO: This assumes that all streams have the same tile count, and every kernel will input and output the same # of tiles.
    //       This assumption no longer holds if kernels are able to do reduction operations. In this case, it should be possible to 
    //       have differing input counts to the same kernel, and differing output counts from the same kernel.

    // Initialize all connections with no count
    for (auto& connection : connections) {
        connection.n_tiles = 0;
    }

    // First pass: Set counts from source streams
    for (size_t i = 0; i < connections.size(); i++) {
        auto& connection = connections[i];
        if (connection.source.is_stream()) {
            auto stream = streams[connection.source.index];
            connection.n_tiles = stream->n_tiles;
        }
    }

    // Keep propagating until no changes are made
    bool changed = true;
    while (changed) {
        changed = false;
        
        // For each kernel
        for (size_t kernel_idx = 0; kernel_idx < kernels.size(); kernel_idx++) {
            // Find all incoming connections with counts
            uint32_t incoming_count = 0;
            bool has_incoming_count = false;
            
            for (const auto& conn : connections) {
                if (conn.dest.is_kernel() && conn.dest.index == kernel_idx && conn.n_tiles > 0) {
                    incoming_count = conn.n_tiles;
                    has_incoming_count = true;
                    break;
                }
            }

            // If we found an incoming count, propagate to all connections for this kernel
            if (has_incoming_count) {
                for (auto& conn : connections) {
                    // Update incoming connections that don't have counts
                    if (conn.dest.is_kernel() && conn.dest.index == kernel_idx && conn.n_tiles == 0) {
                        conn.n_tiles = incoming_count;
                        changed = true;
                    }
                    // Update outgoing connections that don't have counts
                    if (conn.source.is_kernel() && conn.source.index == kernel_idx && conn.n_tiles == 0) {
                        conn.n_tiles = incoming_count;
                        changed = true;
                    }
                }
            }
        }
    }

    // Verify all connections have counts
    for (const auto& connection : connections) {
        if (connection.n_tiles == 0) {
            tt::log_warning("[CURRENT] Connection from {} {} to {} {} has no tile count!",
                connection.source.endpoint_type == Endpoint::EndpointType::Kernel ? "kernel" : "stream",
                connection.source.index,
                connection.dest.endpoint_type == Endpoint::EndpointType::Kernel ? "kernel" : "stream",
                connection.dest.index);
        }
    }
}

} // End namespace current
