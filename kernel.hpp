#pragma once

#include <vector>

#include "common/tt_backend_api_types.hpp"
#include "impl/buffers/buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "common.hpp"

using namespace tt;

namespace current {

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

}
