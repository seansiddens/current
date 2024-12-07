#include "kernel.hpp"

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

}