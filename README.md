# Current

**Current** is a high level parallel programming framework for Tenstorrent accelerators.
It's implemented as a C++ library which allows the user to specify computation kernels in the [Low Level Kernel (LLK)](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html) language which runs on the SFPU.
The framework exposes a streaming programming model and is heavily inspired by projects such as [RaftLib](https://github.com/RaftLib/RaftLib) and the [Brook](https://graphics.stanford.edu/papers/brookgpu/brookgpu.pdf) language.
The stream-based model allows the framework to implicitly extract data- and pipeline-parallelism.
Records in a stream are operated on in parallel by the computation kernel, and connections between kernels allow computation to be pipelined between Tensix cores.
The high level specification of the **Current** program gets transpiled to C++ Metalium host driver code and reader/compute/writer kernels which are automatically scheduled across the Tensix core mesh.

***Disclaimer:*** This project is largely unfinished and will have lots of bugs/inefficiencies when trying to run programs more complicated than the ones described below.

## Example Program

```C++
current::Kernel kernel_a;

// Define ports and set compute kernel.
kernel_a.add_input_port("in0", type);
kernel_a.add_input_port("in1", type);
kernel_a.add_output_port("out0", type);
kernel_a.set_compute_kernel(R"(
    out0 = 2.0f * in0 + in1;
)", false);

// Define streams.
current::Stream source0(generator0_data, count, type);
current::Stream source1(generator1_data, count, type);
current::Stream sink(output_data, count, type);

current::Map({&kernel_a}, {&source0, &source1, &sink});
map.add_connection(&source0, &kernel_a, "in0");
map.add_connection(&source1, &kernel_a, "in1");
map.add_connection(&kernel_a, "out0", &sink);
map.execute();
```

The above snippet showcases how to define a SAXPY kernel in **Current**. 
Streams are essentially just buffers in DRAM, so the corresponding reader kernel that gets generated streams in tiles from the two DRAM buffers:

### Generated Reader Kernel

```C++
// ... [Includes, cicrular buffer initialization, and runtime argument fetching omitted] ...
    while(in0_count < in0_ntiles && in1_count < in1_ntiles) {
        if (in0_count < in0_ntiles) {
            cb_reserve_back(in0, 1);
        }
        if (in1_count < in1_ntiles) {
            cb_reserve_back(in1, 1);
        }
        if (in0_count < in0_ntiles) {
            uint32_t in0_write_ptr = get_write_ptr(in0);
            uint32_t id = in0_tile_offset + in0_count;
            noc_async_read_tile(id, in0_addr_gen, in0_write_ptr);
        }
        if (in1_count < in1_ntiles) {
            uint32_t in1_write_ptr = get_write_ptr(in1);
            uint32_t id = in1_tile_offset + in1_count;
            noc_async_read_tile(id, in1_addr_gen, in1_write_ptr);
        }
        noc_async_read_barrier();
        if (in0_count < in0_ntiles) {
            cb_push_back(in0, 1);
            in0_count++;
        }
        if (in1_count < in1_ntiles) {
            cb_push_back(in1, 1);
            in1_count++;
        }
    }
```
The `cbid_tile_offset` is used in the case where the computation is parallelized across multiple Tensix tiles and the workload is statically partitioned.

Every kernel has defined input and output ports, which can be connected to either streams or other kernels.
In this case, our two input ports are just from streams.

### Generated Compute Kernel

```C++
sfpi_inline void compute() {
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
    for (int i = 0; i < 32; i++) {
        vFloat in0 = dst_reg[0 * 32 + i];
        vFloat in1 = dst_reg[1 * 32 + i];
        vFloat out0;
        out0 = in0 * 2.0 + in1;
        dst_reg[0 * 32 + i] = out0;
    }
}
// ... [Includes and cicrular buffer initialization omitted] ...
void kernel_main() {
    // ... [Kernel argument and SFPU/FPU setup omitted] ...
    // Main processing loop.
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(in0, 1);
        cb_wait_front(in1, 1);
        tile_regs_acquire();
        copy_tile(in0, 0, 0);
        cb_pop_front(in0, 1);
        copy_tile(in1, 0, 1);
        cb_pop_front(in1, 1);
        MATH((sfpi::compute()));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(out0, 1);
        pack_tile(0, out0);
        cb_push_back(out0, 1);
        tile_regs_release();
    }
}
```
.
The computation will wait for data to arrive on it's incoming ports, and then load the data into corresponding vector registers (`in0` is read from CB 0 gets mapped to register `0`, etc).
The programmer specified compute kernel is executed elementwise on the incoming tiles.

## Pipeline Parallelism
The above example showcases how *inter-tile* synchronization is implicitly handled by the framework. 
Input and output circular buffer management as well as SFPU/FPU contention within the compute core is handled automatically.
Additionally, when calling `Map` constructor, an optional `max_parallelization_factor` can be supplied. This argument tells the framework to attempt to automatically parallelize parts of the computation graph across multiple Tensix tiles.
This can be trivially done, since in our model data elements within streams are computed on in parallel, so a static partitioning of the stream records across multiple Tensix tiles is a trivial (naive) approach.

The framework also allows for implict `intra-tile` synchronization via kernel-to-kernel connections. 
This further allows for further pipeline parallelism to be extracted from programs. The main difference with regards to kernel generation for these type of programs is we must generate writer and reader kernels which correctly implement a producer/consumer relationship:

### Generated Writer (Producer) Kernel

```C++
    for(uint32_t i = 0; i < out0_ntiles; i++) {
        noc_semaphore_wait(out0_sender_semaphore_addr_ptr, 1);
        cb_wait_front(out0, 1);
        uint32_t out0_read_ptr = get_read_ptr(out0);
        uint32_t out0_receiver_read_ptr = get_read_ptr(0);
        uint64_t out0_receiver_noc_addr = get_noc_addr(out0_receiver_noc_x, out0_receiver_noc_y, out0_receiver_read_ptr);
        noc_async_write(out0_read_ptr, out0_receiver_noc_addr, 2048);
        noc_semaphore_set(out0_sender_semaphore_addr_ptr, 0);
        noc_semaphore_set_remote(out0_l1_valid_value_addr, out0_receiver_semaphore_noc_addr);
        cb_pop_front(out0, 1);
    }
```

### Generated Reader (Consumer) Kernel

```C++
    while(in0_count < in0_ntiles) {
        cb_reserve_back(in0, 1);
        noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
        cb_push_back(in0, 1);
        in0_count++;
    }
```

If our program had a connection between kernel A and kernel B, and these kernels were mapped to Tensix cores 0 and 1 respectively, then the above writer kernel would be generated to run on core 0 and the above reader would be generated to run on core 1.


## Gather Streams
The features described thus far would only allow for the specification of data paralell operations across contiguous memory.
A useful construct would be that of a **gather stream**, which is a stream of indices into an associated data buffer. 
This abstraction would essentially allow for random access.


```C++
    while(in0_count < in0_ntiles) {
        if (in0_count < in0_ntiles) {
            cb_reserve_back(in0_0, 1);
            cb_reserve_back(in0_1, 1);
        }
        if (in0_count < in0_ntiles) {
            uint32_t id = in0_tile_offset + in0_count;
            noc_async_read_tile(id, in0_addr_gen, in0_indices_write_ptr);
        }
        noc_async_read_barrier();
        if (in0_count < in0_ntiles) {
            uint32_t in0_0_write_ptr = get_write_ptr(in0_0);
            uint32_t in0_1_write_ptr = get_write_ptr(in0_1);
            uint32_t index;
            for (int i = 0; i < 1024; i++) {
                index = *(((uint32_t *)in0_indices_write_ptr) + (i * 2 + 0)) * 32;
                uint32_t in0_0_offset = i * 2;
                noc_async_read(in0_data_dram_noc_addr + index, in0_0_write_ptr + in0_0_offset, 2);
                index = *(((uint32_t *)in0_indices_write_ptr) + (i * 2 + 1)) * 32;
                uint32_t in0_1_offset = i * 2;
                noc_async_read(in0_data_dram_noc_addr + index, in0_1_write_ptr + in0_1_offset, 2);
            }
        }
        noc_async_read_barrier();
        if (in0_count < in0_ntiles) {
            cb_push_back(in0_0, 1);
            cb_push_back(in0_1, 1);
            in0_count++;
        }
    }
```

The above snippet is the generated reader kernel for a gather stream.
The Tenstorrent architecture is not optimzed for fine-grained non-contiguous access from DRAM, and I found that attempting to read DRAM addresses that were not 32 byte aligned failed.
The naive solution I implemented was to simply pad the data elements such that every record would be at a 32 byte boundary.
The reader kernel first reads in a tile of indices, and then each index is used to fetch an element from DRAM and is written to a CB.

This approach is obviously incredibly inefficient, so I included an option to tell the framework to attempt to store the entire data buffer in SRAM. This removes the need for inefficient memory padding or unneeded DRAM accesses.


## Future Work
The goal of this project is to act as a proof of concept for how to make Tenstorrent architectures easier to program (similar to how the Brook work tried to do the same for GPUs).
The framework automatically handles inter- and intra-Tensix synchronization to a degree, but is still quite limited in it's programming model.
Future work could build off of this framework to expose a more intuitive and flexible interface and/or DSL.
Additionally, exposing the SRAM scratchpad memories to exploit data re-use should be further explored.

Please email me (`seansiddens[at]gmail[dot]com`) or DM me on the Tenstorrent discord (`@.sren`) if you have any further questions/suggestions.


