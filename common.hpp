#pragma once

#include "common/bfloat16.hpp"

// TODO: Need to determine when/how these will be set. 
// I think I might want to keep this constant for a given runtime program graph.
// i.e have the granularity of tokens/messages within streams always be that of a tile.
// Eventually will need to parameterize by the type.
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_SIZE = TILE_WIDTH * TILE_HEIGHT;
constexpr uint32_t TILE_SIZE_BYTES = TILE_SIZE * sizeof(bfloat16);
constexpr uint32_t TILES_PER_CB = 4;
constexpr uint32_t IN_CB_START = 0;
constexpr uint32_t OUT_CB_START = 16;
constexpr uint32_t MAX_INPUT_PORTS = 16;
constexpr uint32_t MAX_OUTPUT_PORTS = 16;
// constexpr uint32_t INVALID = 0;
// constexpr uint32_t VALID = 1;