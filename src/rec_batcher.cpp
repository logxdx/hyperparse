#include "rec_batcher.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "core/kernels/kernels.h"
#include "core/recognition/ctc_decode.h"
#include "core/recognition/rec_geometry.h"
#include "script.h"

namespace hp {

bool RecBatcher::load(const std::string &plan, const std::string &dict) {
  engine_ = std::make_unique<hp::core::engine::TrtEngine>(plan);
  if (!engine_->load()) return false;
  nvinfer1::Dims d{};
  d.nbDims = 4;
  d.d[0] = 1;
  d.d[1] = 3;
  d.d[2] = kH;
  d.d[3] = 320;
  int seq = 0;
  engine_->probe_output_dims(d, seq, classes_);
  labels_.push_back("blank");
  if (!hp::core::recognition::load_label_dict(dict, labels_)) return false;
  if ((int)labels_.size() != classes_) throw std::runtime_error("rec dict/model class mismatch");
  const std::vector<uint8_t> mask = non_cjk_class_mask(labels_);
  d_non_cjk_ = hp::core::CudaPtr<uint8_t>(mask.size());
  CUDA_CHECK(cudaMemcpy(d_non_cjk_.get(), mask.data(), mask.size(), cudaMemcpyHostToDevice));
  // Largest single batch: the column budget, or one line at max width.
  const size_t max_cols = std::max(kColumnBudget, kMaxW);
  d_out_ = hp::core::CudaPtr<float>(max_cols / 8 * (size_t)classes_);
  return true;
}

std::vector<std::pair<std::string, float>> RecBatcher::run(const hp::core::GpuImage &img,
                                                           const std::vector<hp::core::Box> &boxes,
                                                           cudaStream_t s, bool no_cjk) {
  const int n = (int)boxes.size();
  std::vector<std::pair<std::string, float>> out(n);
  if (n == 0) return out;
  std::vector<int> w(n), order(n);
  for (int i = 0; i < n; ++i) {
    const int rw = std::min(hp::core::recognition::rec_input_width(boxes[i], kH), kMaxW);
    w[i] = std::max(48, (rw + 31) / 32 * 32);  // engine profile min width 48
  }
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, {}, [&](int i) { return w[i]; });

  struct Batch { int beg, end, W; size_t in_off, row_off; int T; };
  std::vector<Batch> bs;
  size_t in_total = 0, rows_total = 0;
  for (int beg = 0; beg < n;) {
    int end = beg, W = 0;
    while (end < n && end - beg < kMaxBatch) {
      const int nw = std::max(W, w[order[end]]);
      if (end > beg && (end - beg + 1) * nw > kColumnBudget) break;
      W = nw;
      ++end;
    }
    bs.push_back({beg, end, W, in_total, rows_total, W / 8});
    in_total += (size_t)(end - beg) * 3 * kH * W;
    rows_total += (size_t)(end - beg) * (W / 8);
    beg = end;
  }
  if (in_total > in_cap_) { d_in_.reset(in_total); in_cap_ = in_total; }
  if (rows_total > rows_cap_) {
    d_idx_.reset(rows_total); d_score_.reset(rows_total);
    h_idx_.reset(rows_total); h_score_.reset(rows_total);
    rows_cap_ = rows_total;
  }
  // One launch: every line of every batch, straight into its batch's input tensor.
  crops_.clear();
  for (const auto &b : bs)
    for (int j = b.beg; j < b.end; ++j)
      crops_.add_rec_line(boxes[order[j]], d_in_.get() + b.in_off + (size_t)(j - b.beg) * 3 * kH * b.W, b.W, kH);
  crops_.launch(static_cast<const uint8_t *>(img.data), img.step, img.cols, img.rows, s);

  for (auto &b : bs) {
    const int cnt = b.end - b.beg;
    nvinfer1::Dims d{};
    d.nbDims = 4;
    d.d[0] = cnt;
    d.d[1] = 3;
    d.d[2] = kH;
    d.d[3] = b.W;
    engine_->bind_io(d_in_.get() + b.in_off, d_out_.get());
    if (!engine_->infer_dynamic(d, s)) throw std::runtime_error("rec execute failed");
    b.T = engine_->get_output_dims().d[1];
    if (b.T > b.W / 8) throw std::runtime_error("rec: unexpected sequence length");
    hp::core::kernels::cuda_argmax(d_out_.get(), d_idx_.get() + b.row_off, d_score_.get() + b.row_off, cnt, b.T,
                                    classes_, s, no_cjk ? d_non_cjk_.get() : nullptr);
  }
  // Rows are laid out per batch at stride W/8 per line; copy the whole region once.
  CUDA_CHECK(cudaMemcpyAsync(h_idx_.get(), d_idx_.get(), rows_total * sizeof(int), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(h_score_.get(), d_score_.get(), rows_total * sizeof(float), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  for (const auto &b : bs)
    for (int j = b.beg; j < b.end; ++j)
      out[order[j]] = hp::core::recognition::ctc_greedy_decode(h_idx_.get() + b.row_off + (size_t)(j - b.beg) * b.T,
                                                               h_score_.get() + b.row_off + (size_t)(j - b.beg) * b.T,
                                                               b.T, labels_);
  batches_run += (int)bs.size();
  lines_run += n;
  return out;
}

}  // namespace hp
