#pragma once
// Documents through the loaded engine. One long-lived pipeline serves every
// document in flight, so the GPU does not idle between documents:
//
//   caller thread (per document)  open the PDF for its page count, publish it in a
//        │                        free document slot of the pool, wait for its pages
//   page workers (RenderPool)     text layer -> route -> render into a pinned slot
//        │  Done (pipe)
//   GPU thread                    per page of any document, alternating two LayoutLanes:
//        │                          H2D upload -> multi_resize -> D-FINE (lane stream)
//        │                          OCR stage on the OCR stream (ocr_stage.h)
//        │                          lane sync -> copy formula crops, grey page for
//        │                          LayoutNet (converted on the GPU) -> slot freed
//        │  PageWork (queue), pages with formula crops first through:
//   formula thread                formula decode, batched across queued pages (of
//        │                        any document)
//   assembly threads (3)          LayoutNet tables, reading order + Markdown
//                                 (OrderNet) -> sink, a page each, in parallel
//
// Back-pressure is the slot ring: a page is only submitted to the workers when a
// slot is free, and a slot is freed as soon as the GPU no longer reads it.
//
// Sharing between documents: free slots, and the GPU among rendered pages, go
// first to the document with the fewest pages left (ties: the earliest), so a
// short document overtakes a long one instead of queueing behind all its pages;
// every third slot and every third GPU page go to the earliest instead, so a
// long document keeps at least a third under a stream of short ones.
//
// Why one pipeline for all documents: run one document at a time end to end and
// every document pays render latency (text layer + raster, ~21 ms for a page)
// with the GPU idle, then GPU time with the workers idle, then assembly with
// both idle. Measured on 300 one-page documents (RTX 4050 laptop, 8 workers,
// 8 requests in flight): 49 ms per document serialized (first page rendered at
// 21 ms, GPU done at 44 ms), so 20 pages/s, against 44.5 pages/s for the same
// pages as one PDF. The probe, the memfd hand-off and the thread start cost
// 0.1-0.2 ms together. Overlapped: 43-44 pages/s, the one-PDF rate.
// The same documents with a 302-page and a 100-page PDF among them: short
// requests p50/p90/p99 156/332/637 ms (one document at a time: 350/960/7114 ms,
// stuck behind the long ones); the whole mix 20.0 s instead of 28.5 s.
//
// Choices, with evidence:
//  * Two lanes: D-FINE for page i+1 overlaps OCR for page i. GPU utilization was
//    100% on the IDL scan run, but 1 lane vs 2 has not been A/B tested in
//    isolation. A lane whose page has no successor ready is finished at once
//    (before the GPU thread blocks), so a document's last page never waits for
//    another document's first.
//  * Formulas on their own thread and stream: decoding is autoregressive (25.2 ms
//    per formula with the FP16 step) and would stall the lanes. Measured on 60
//    paper pages with 98 formulas: 25.1 pages/s without formulas, 11.8 with.
//    Batching across pages fills the decoder's 8 sequence slots
//    (FormulaRecognizer::kMaxBatch); its gain over per-page batches is unmeasured.
//    The formula thread is apart from the assembly threads, so a cohort being
//    decoded holds up only pages with formulas. A second decoder on a second
//    thread gained 3% on arxiv100 (17.2 -> 17.8 pages/s): one decoder already
//    keeps the GPU busy, so there is one.
//  * Assembly on several threads: one thread was 44-48% busy on merged300 (LayoutNet
//    13 ms per table page, OrderNet + Markdown 2.6 ms per page) and would bound
//    the pipeline once the GPU side is faster. The CPU parts (features, decode,
//    table clean-up, Markdown) run in parallel; LayoutNet and OrderNet each have
//    one context, and their forward() calls take turns. A document's sink is
//    never called twice at once.
//  * Slots: 2 × workers + 2, so every worker can render ahead while both lanes
//    hold a page.
//
// Errors are per document. A page that fails to render arrives as
// PageResult::ok == false. Anything else that goes wrong with one of a
// document's pages (corrupt metadata, a GPU stage throwing, its sink throwing)
// fails that document only: it stops getting slots, its jobs already submitted
// are drained (the pool guarantees one result per job), its undelivered pages
// are dropped, and parse() rethrows the first error once nothing of the
// document is left in the pipeline. Other documents carry on. A broken pool
// (dead or stalled) fails every document in flight; the next document then
// fails fast if the pool is still dead.

#include <cstdint>
#include <memory>
#include <span>

#include "model_set.h"
#include "parse_types.h"
#include "render_pool.h"

namespace hp {

class PagePipeline {
 public:
  // Starts the GPU, formula and `assembly_threads` assembly threads. `models` and
  // `pool` must outlive it.
  PagePipeline(ModelSet &models, RenderPool &pool, int assembly_threads);
  // No parse() may be running.
  ~PagePipeline();
  PagePipeline(const PagePipeline &) = delete;
  PagePipeline &operator=(const PagePipeline &) = delete;

  // Parse one document; blocks until every page has been delivered to `sink`
  // (or the document failed). Thread-safe: up to pool.documents() calls run at
  // once and share the GPU; more wait for a document slot.
  ParseStats parse(std::span<const uint8_t> pdf, const ParseOptions &opt, const PageSink &sink);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hp
