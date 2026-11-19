#pragma once
// One document through the loaded engine:
//
//   page workers (RenderPool)     text layer -> route -> render into a pinned slot
//        │  Done (pipe)
//   GPU thread (caller)           per page, alternating two LayoutLanes:
//        │                          H2D upload -> multi_resize -> D-FINE (lane stream)
//        │                          OCR stage on the OCR stream (ocr_stage.h)
//        │                          lane sync -> copy formula crops -> slot freed
//        │  PageWork (queue)
//   assembly thread               formula decode, batched across queued pages
//                                 -> reading order + Markdown -> PageSink
//
// Back-pressure is the slot ring: a page is only submitted to the workers when a
// slot is free, and a slot is freed as soon as the GPU no longer reads it.
// Choices, with evidence:
//  * Two lanes: D-FINE for page i+1 overlaps OCR for page i. GPU utilization was
//    100% on the IDL scan run, but 1 lane vs 2 has not been A/B tested in
//    isolation.
//  * Formulas on their own thread and stream: decoding is autoregressive (25.2 ms
//    per formula with the FP16 step) and would stall the lanes. Measured on 60
//    paper pages with 98 formulas: 25.1 pages/s without formulas, 11.8 with.
//    Batching across pages fills the decoder's 8 sequence slots
//    (FormulaRecognizer::kMaxBatch); its gain over per-page batches is unmeasured.
//  * Slots: 2 × workers + 2, so every worker can render ahead while both lanes
//    hold a page.
//
// Errors: if any stage throws, submission stops, every outstanding page job
// is drained (the pool guarantees one result per job), the GPU streams are
// synchronized and the assembly thread is joined. Only then is the exception
// rethrown, so the pool and models stay reusable for the next document.

#include <cstdint>
#include <span>

#include "model_set.h"
#include "parse_types.h"
#include "render_pool.h"

namespace hp {

// `pool` must be idle (no outstanding jobs) and `models` loaded. Blocks until every
// page has been delivered to `sink`.
ParseStats parse_document(ModelSet &models, RenderPool &pool, std::span<const uint8_t> pdf, const ParseOptions &opt,
                          const PageSink &sink);

}  // namespace hp
