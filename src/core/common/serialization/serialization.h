#pragma once

// JSON serialization, split by concern. This umbrella preserves the
// original single-include surface for every call site; each piece stays
// header-only inline (hot path across 20 include sites).
//   serialization_primitives.h  escaping + scalar/box writers
//   serialization_items.h       per-item and array writers
//   serialization_blocks.h      paragraph-level blocks aggregate
//   serialization_emit.h        assign_layout_ids + response emitters
#include "core/common/serialization/serialization_primitives.h"

#include "core/common/serialization/serialization_items.h"

#include "core/common/serialization/serialization_blocks.h"

#include "core/common/serialization/serialization_emit.h"
