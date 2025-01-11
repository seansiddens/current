#include <vector>

#include "common.hpp"

#include "kernel.hpp"
#include "stream.hpp"


#pragma once

namespace current {

class Map {
  public:
    Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams, uint32_t max_parallelization_factor=1, uint32_t tiles_per_cb=1);
    ~Map();
    void add_connection(Kernel *src, std::string src_out, Kernel *dst, std::string dst_in);
    void add_connection(Stream *src, Kernel *dst, std::string dst_in);
    void add_connection(Kernel *src, std::string src_out, Stream *dst);
    void execute();
    void generate_device_kernels();
    void check_connections();
    void propagate_counts();
    std::vector<uint32_t> read_stream(Stream *stream);
    std::vector<uint32_t> read_gather_stream(Stream *stream, bool read_data);
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

    uint32_t max_parallelization_factor;
    uint32_t tiles_per_cb;
    std::optional<Runtime> runtime;
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
};


} 