#pragma once

#include <vector>
#include <filesystem>

#include "common/bfloat16.hpp"
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
    Stream(const std::vector<uint32_t>& initial_data, size_t num_elements, tt::DataFormat data_format_) {
        // std::cout << "initial_data.size(): " << initial_data.size() << "\n";
        // std::cout << "num_elements: " << num_elements << "\n";
        // std::cout << "el size: " << tt::datum_size(data_format_) << "\n";
        // assert(initial_data.size() * 4 == num_elements * tt::datum_size(data_format_) && "Stream data size does not match number of elements!");
        n_elements = num_elements;
        host_data = initial_data;
        this->element_size = tt::datum_size(data_format_);
        this->format = data_format_;
        this->n_tiles = static_cast<uint32_t>(std::ceil(num_elements / static_cast<double>(TILE_SIZE)));
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
    tt::DataFormat format;

    friend class Map;
};
class GatherStream : public Stream {
  public: 
    GatherStream(const std::vector<uint32_t>& data_buffer, 
                 tt::DataFormat data_format,
                 uint32_t n_elements_,
                 const std::vector<uint32_t>& index_data,
                 bool use_sram=false,
                 uint8_t accesses_per_token=1);
    [[nodiscard]] bool is_gather_stream() const override { return true; }

    static GatherStream CreateStencil(const std::vector<uint32_t>& data,
                                      tt::DataFormat format,
                                      bool use_sram = false);


  private:
    std::vector<uint32_t> data_buffer;
    std::shared_ptr<tt_metal::Buffer> data_buffer_device;
    uint32_t data_buffer_address;
    tt_metal::CoreCoord data_buffer_noc_coordinates;
    tt::DataFormat data_format;
    uint32_t data_n_elements;
    uint32_t data_n_tiles;
    uint8_t accesses_per_token;
    bool use_sram;

    friend class Map;
};

} // End namespace current.
