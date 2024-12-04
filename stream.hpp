#pragma once

#include <vector>
#include <filesystem>

#include "impl/buffers/buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "common.hpp"

using namespace tt;

namespace current {

const std::filesystem::path GENERATED_KERNELS_PATH = "tt_metal/programming_examples/personal/current/kernels/generated";

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

// Wrapper around a DRAM buffer. Used as source and dest of stream data for kernels.
class Stream {
  public: 
    Stream(const std::vector<uint32_t>& initial_data, size_t num_elements, tt::DataFormat data_format) {
        assert(initial_data.size() * 4 == num_elements * tt::datum_size(data_format) && "Stream data size does not match number of elements!");
        n_elements = num_elements;
        host_data = initial_data;
        this->element_size = tt::datum_size(data_format);
        this->data_format = data_format;
        this->n_tiles = std::ceil(n_elements / TILE_SIZE);
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
    size_t n_elements;   // Number of elements/tokens this stream will produce.
    size_t element_size; // Size (in bytes) of each element
    uint32_t n_tiles;    // Total # of 32x32 tiles this stream will produce.
    tt::DataFormat data_format;
};

class Kernel {
  public: 
    Kernel() = default;

    struct Port {
        std::string name;
        tt::DataFormat data_format;
        tt_metal::CBHandle cb; // TODO: Do we want ports to have ownership of CBs?
        // L1 buffer for handling incoming mailbox messages.
        // Only used for input ports whose incoming connections are another kernel.
        std::shared_ptr<tt_metal::Buffer> mailbox; 
    };

    void add_input_port(const std::string& name, tt::DataFormat data_format);
    void add_output_port(const std::string& name, tt::DataFormat data_format);
    uint32_t num_input_ports() const;
    uint32_t num_output_ports() const;

    void set_compute_kernel(const std::string& code, bool do_matmul = false) {
        size_t last = code.find_last_not_of(" \t\n\r");
        sfpi_kernel_string = (last != std::string::npos) ? code.substr(0, last + 1) + "\n\n" : "";
        this->do_matmul = do_matmul;
    }

    uint32_t get_input_port_index(std::string port_name) const {
        for (size_t i = 0; i < input_ports.size(); i++) {
            if (input_ports[i].name == port_name) {
                return i;
            }
        }
        return -1;
    }

    Port get_input_port(std::string port_name) const {
        for (size_t i = 0; i < input_ports.size(); i++) {
            if (input_ports[i].name == port_name) {
                return input_ports[i];
            }
        }
        assert(false && "Input port not found!");
    }

    Port get_output_port(std::string port_name) const {
        for (size_t i = 0; i < output_ports.size(); i++) {
            if (output_ports[i].name == port_name) {
                return output_ports[i];
            }
        }
        assert(false && "Output port not found!");
    }

    uint32_t get_output_port_index(std::string port_name) const {
        for (size_t i = 0; i < output_ports.size(); i++) {
            if (output_ports[i].name == port_name) {
                return i;
            }
        }
        return -1;
    }


    // TODO: For each input port, we need need a CB to move data from NOC to compute.
    // If the kernel is a generator, each input port will be reading from a DRAM buffer.
    // If our input/output ports are connected to other kernels, need to determine how to do
    // the pipelining.
    std::vector<Port> input_ports;
    std::vector<Port> output_ports;
    std::vector<CoreCoord> core_spec; // Where this kernel will be placed.
    tt_metal::KernelHandle reader_kernel;
    tt_metal::KernelHandle compute_kernel;
    tt_metal::KernelHandle writer_kernel;
    std::filesystem::path generated_reader_kernel_path;
    std::filesystem::path generated_compute_kernel_path;
    std::filesystem::path generated_writer_kernel_path;
    std::string sfpi_kernel_string;
    bool do_matmul = false;
    // TODO: This only allows for one sender and receiver per kernel.
    // Eventually would want to support multiple senders and receivers per kernel
    // e.g when we have multiple output ports, each participatnig in producer/consumer.
    // Or if we have single producer and multiple consumers (and vice versa).
    uint32_t sender_semaphore_id;
    uint32_t receiver_semaphore_id;
    uint32_t l1_valid_value_semaphore_id;
};

class Map {
  public:
    Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams);
    ~Map();
    void add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in);
    void add_connection(Stream *src, Kernel *dst, std::string dst_in);
    void add_connection(Kernel *src, std::string src_out, Stream *dst);
    void execute();
    void generate_device_kernels();
    void check_connections();
    std::vector<uint32_t> read_stream(Stream *stream);
    void parallelize(std::vector<CoreCoord> &cores);

    // Visualize the work graph.
    // For PNG
    // dot -Tpng filename.dot -o filename.png
    // For SVG (better for web/documentation)
    // dot -Tsvg filename.dot -o filename.svg
    // For PDF (better for papers/presentations)
    // dot -Tpdf filename.dot -o filename.pdf
    void export_dot(const std::string& filename) const;

  private:
    struct Runtime {
        tt_metal::Device *device;
        tt_metal::Program program;
        uint32_t num_cores;
        uint32_t num_cores_x;
        uint32_t num_cores_y;
        tt_metal::CoreRangeSet core_set;
    };

    // Represents a connection endpoint (either kernel or stream)
    struct Endpoint {
        enum class EndpointType { Kernel, Stream };
        EndpointType endpoint_type;
        size_t index;      // Index into either kernels or streams vector
        std::string port;  // Port name (only valid for kernels)

        bool is_kernel() const { return endpoint_type == EndpointType::Kernel; }
        bool is_stream() const { return endpoint_type == EndpointType::Stream; }
    };

    // Represents a directed connection between endpoints
    struct Connection {
        Endpoint source;
        Endpoint dest;
        uint32_t n_tiles;
    };

    Runtime runtime;
    std::vector<Kernel *> kernels;
    std::vector<Stream *> streams;
    std::vector<Connection> connections;

    // // Entry <port_out, port_in> at [i][j] represents a connection from kernel i's output port port_out to kernel j's input port port_in.
    // std::vector<std::vector<std::pair<std::string, std::string>>> port_map;
    // // Entry <port_name> at [i][j] represents a connection from stream i to kernel j's input port port_name.
    // std::vector<std::vector<std::string>> stream_port_map;

    std::vector<Connection> get_incoming_connections(Kernel *kernel);
    std::vector<Connection> get_outgoing_connections(Kernel *kernel);

    size_t get_kernel_index(Kernel *kernel) {
        auto it = std::find(kernels.begin(), kernels.end(), kernel);
        assert(it != kernels.end() && "Kernel not found in kernels vector");
        return it - kernels.begin();
    }

    size_t get_stream_index(Stream *stream) {
        auto it = std::find(streams.begin(), streams.end(), stream);
        assert(it != streams.end() && "Stream not found in streams vector");
        return it - streams.begin();
    }

    void add_connection(const Endpoint& src, const Endpoint& dst) {
        connections.push_back({src, dst});
    }

    void generate_reader_device_kernel(Kernel *kernel, std::vector<Connection> incoming_connections);
    void generate_compute_device_kernel(Kernel *kernel, std::vector<Connection> incoming_connections, std::vector<Connection> outgoing_connections);
    void generate_writer_device_kernel(Kernel *kernel, std::vector<Connection> outgoing_connections);
    bool has_incoming_connection(Kernel *kernel);
    void propagate_counts();
};

} // End namespace current.
