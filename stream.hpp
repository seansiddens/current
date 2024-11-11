#pragma once

#include <utility>
#include <unordered_map>
#include <vector>

#include "impl/buffers/buffer.hpp"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

using namespace tt;

namespace stream {

using PortName = std::string;

// Wrapper around a DRAM buffer. Used as source and dest of stream data for kernels.
class Stream {
  public: 
    Stream(std::vector<uint32_t> inital_data, size_t num_elements, size_t element_size) {
        n_elements = num_elements;
        host_data = inital_data;
        this->element_size = element_size;
    }

  private:
    friend class Map;
    // Corresponding host data for the buffer. 
    // If this is a source, then the host will initialize this data and the runtime will copy it to the device.
    // If this is a sink, then the runtime will read data from the device into this buffer for the host to read.
    // TODO: Might not have to actually do it this way. If instead we have like a 
    // writeStream() and readStream() function, then the Stream class won't actually hold any data (this is what Brook does).
    std::vector<uint32_t> host_data;  
    std::shared_ptr<tt_metal::Buffer> device_buffer;
    uint32_t device_buffer_address;
    tt_metal::CoreCoord device_buffer_noc_coordinates;
    size_t n_elements;
    size_t element_size;
};

class Kernel {
  public: 
    Kernel() = default;

    void add_input_port(const std::string& name, tt::CB cb);
    void add_output_port(const std::string& name, tt::CB cb);
    uint32_t num_input_ports() const;
    uint32_t num_output_ports() const;

    // TODO: For each input port, we need need a CB to move data from NOC to compute.
    // If the kernel is a generator, each input port will be reading from a DRAM buffer.
    // If our input/output ports are connected to other kernels, need to determine how to do
    // the pipelining.
    std::unordered_map<std::string, tt::CB> input_ports;
    std::unordered_map<std::string, tt::CB> output_ports;
};

class Map {
  public:
    Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams);
    void add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in);
    void add_connection(Stream *src, Kernel *dst, std::string dst_in);
    void add_connection(Kernel *src, std::string src_out, Stream *dst);
    void execute();
    void generate_device_kernels();

  private:
    struct Runtime {
        tt_metal::Device *device;
        tt_metal::Program program;
        uint32_t num_cores;
        uint32_t num_cores_x;
        uint32_t num_cores_y;
        std::set<tt_metal::CoreRange> core_set;
    };

    // Represents a connection endpoint (either kernel or stream)
    struct Endpoint {
        enum class Type { Kernel, Stream };
        Type type;
        size_t index;      // Index into either kernels or streams vector
        std::string port;  // Port name (only valid for kernels)

        bool is_kernel() const { return type == Type::Kernel; }
        bool is_stream() const { return type == Type::Stream; }
    };

    // Represents a directed connection between endpoints
    struct Connection {
        Endpoint source;
        Endpoint dest;
    };

    Runtime runtime;
    std::vector<Kernel *> kernels;
    std::vector<Stream *> streams;
    std::vector<Connection> connections;

    // // Entry <port_out, port_in> at [i][j] represents a connection from kernel i's output port port_out to kernel j's input port port_in.
    // std::vector<std::vector<std::pair<std::string, std::string>>> port_map;
    // // Entry <port_name> at [i][j] represents a connection from stream i to kernel j's input port port_name.
    // std::vector<std::vector<std::string>> stream_port_map;

    size_t get_kernel_index(Kernel *kernel) {
        return std::find(kernels.begin(), kernels.end(), kernel) - kernels.begin();
    }

    size_t get_stream_index(Stream *stream) {
        return std::find(streams.begin(), streams.end(), stream) - streams.begin();
    }

    void add_connection(const Endpoint& src, const Endpoint& dst) {
        connections.push_back({src, dst});
    }

    void generate_reader_device_kernel(Kernel *kernel);
    void generate_writer_device_kernel(Kernel *kernel);
    void generate_compute_device_kernel(Kernel *kernel);
    bool has_incoming_connection(Kernel *kernel);
};

} // End namespace stream.