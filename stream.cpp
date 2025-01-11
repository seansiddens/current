#include "stream.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>

#include "common/tt_backend_api_types.hpp"
#include "detail/tt_metal.hpp"
#include "host_api.hpp"
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "common/work_split.hpp"
#include "tt_metal/impl/device/device.hpp"


#include "common.hpp"

namespace current {


inline int get_tt_npu_clock(tt::tt_metal::Device *device) {
    return tt::Cluster::instance().get_device_aiclk(device->id());
}

// GatherStream GatherStream::CreateStencil(const std::vector<uint32_t> &data, tt::DataFormat format, bool use_sram) {
// }


GatherStream::GatherStream(const std::vector<uint32_t>& data_buffer, 
                 tt::DataFormat data_format,
                 uint32_t n_elements_,
                 const std::vector<uint32_t>& index_data,
                 bool use_sram,
                 uint8_t accesses_per_token)
                 : Stream(index_data, index_data.size(), tt::DataFormat::UInt32), data_format(data_format), use_sram(use_sram), accesses_per_token(accesses_per_token) {
    assert(TILE_SIZE % accesses_per_token == 0 && "Accesses per token must evenly divide TILE SIZE! (1024)\n");
    this->data_n_elements = n_elements_;
    if (use_sram) {
        std::cout << "Using SRAM for the GatherStream!\n";
        // If we are placing the entire data buffer in the L1, padding is not needed.
        this->data_buffer = data_buffer;
    } else {
        if (data_format == tt::DataFormat::Float16_b) {
            // Accesses need to be at 32 byte boundaries, so we must pad our data every 16 b16 elements.
            std::vector<bfloat16> in(data_n_elements * 16, bfloat16(0.0F));
            std::vector<bfloat16> initial_data = unpack_uint32_vec_into_bfloat16_vec(data_buffer);
            std::cout << "Initial data size: " << initial_data.size() << "\n";
            for (size_t i = 0; i < in.size(); i++) {
                in[i] = initial_data[i / 16];
            }
            this->data_buffer = pack_bfloat16_vec_into_uint32_vec(in);
            std::cout << "Scaled data buffer: " << this->data_buffer.size() << "\n";
        } else {
            assert(false && "Unsupported data type for gather stream!\n");
        }
        // auto factor = 32 / datum_size(data_format);
        // this->data_buffer.resize(data_buffer.size() * 8); // Every data element needs to be 32-byte aligned (8 u32s).
        // for (size_t i = 0; i < this->data_buffer.size(); i++) {
        //     // size_t idx = static_cast<uint32_t>(i) & ~7U; // Clears 3 LSBs, rounding down to nearest multiple of 8.
        //     size_t idx = static_cast<uint32_t>(i / 8U);
        //     this->data_buffer[i] = data_buffer[idx];
        // }
        // auto foo = unpack_uint32_vec_into_bfloat16_vec(this->data_buffer);
        // for (size_t i = 0; i < foo.size(); i++) {
        //   std::cout << i << ": " << foo[i].to_float() << "\n";
        // }
        // this->data_n_tiles = static_cast<uint32_t>(std::ceil(this->data_n_elements / static_cast<double>(TILE_SIZE)));
        // std::cout << "Gather data n_elements: " << this->data_n_elements << "\n";
        // std::cout << "Gather data n_tiles: " << this->data_n_tiles << "\n";
    }
}


} // End namespace current
