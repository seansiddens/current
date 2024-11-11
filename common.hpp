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