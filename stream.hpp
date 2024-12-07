#pragma once

#include <vector>
#include <filesystem>

#include "common/tt_backend_api_types.hpp"
#include "impl/buffers/buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "common.hpp"

#include "kernel.hpp"

using namespace tt;

namespace current {

const std::filesystem::path GENERATED_KERNELS_PATH = "sources/examples/current/kernels/generated";

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

    virtual ~Stream() = default;

    [[nodiscard]] virtual bool is_gather_stream() const { return false; }

  protected:
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

    friend class Map;
};

class GatherStream : public Stream {
  public: 
    GatherStream(const std::vector<uint32_t>& data_buffer, 
                 tt::DataFormat data_format,
                 const std::vector<uint32_t>& index_data)
                 : Stream(index_data, index_data.size(), tt::DataFormat::UInt32), data_format(data_format) {
        // auto n_elements = (data_buffer.size() * sizeof(data_buffer[0])) / datum_size(this->data_format);
        this->data_buffer.resize(data_buffer.size() * 8); // Every data element needs to be 32-byte aligned (8 u32s).
        for (size_t i = 0; i < this->data_buffer.size(); i++) {
            // size_t idx = static_cast<uint32_t>(i) & ~7U; // Clears 3 LSBs, rounding down to nearest multiple of 8.
            size_t idx = static_cast<uint32_t>(i / 8U);
            this->data_buffer[i] = data_buffer[idx];
            // std::cout << "data_buf " << i << ": " << this->data_buffer[i] << "\n";
        }
    }

    [[nodiscard]] bool is_gather_stream() const override { return true; }


  private:
    std::vector<uint32_t> data_buffer;
    std::shared_ptr<tt_metal::Buffer> data_buffer_device;
    uint32_t data_buffer_address;
    tt_metal::CoreCoord data_buffer_noc_coordinates;
    tt::DataFormat data_format;

    friend class Map;
};

class Map {
  public:
    Map(std::vector<Kernel *> kernels, std::vector<Stream *> streams, uint32_t max_parallelization_factor);
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
    void propagate_counts();
};

} // End namespace current.
