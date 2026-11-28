#pragma once
#include <cstdint>
struct FusedQuantPayloadTilingData { uint32_t batchSize; uint32_t hiddenSize; uint32_t topK; uint32_t rowBytes; };
