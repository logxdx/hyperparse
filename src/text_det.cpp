#include "text_det.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hp {

using hp::core::Box;
using hp::core::CudaHostPtr;
using hp::core::CudaPtr;
using hp::core::kernels::GpuDetBox;
using hp::core::kernels::kMaxGpuComponents;

namespace {

constexpr float kMinUnclippedSide = 5.0f;  // PaddleDet
constexpr float kMaxExpand = 24.0f;        // also bounds the JFA jump range
constexpr float kMinExpand = 2.0f;

// Same ordering as order_quad_tl_tr_br_bl (stable sort by x, then
// left pair -> tl/bl by y, right pair -> tr/br by y).
void order_quad(float xs[4], float ys[4]) {
  for (int i = 1; i < 4; ++i) {
    const float kx = xs[i], ky = ys[i];
    int j = i - 1;
    while (j >= 0 && xs[j] > kx) {
      xs[j + 1] = xs[j];
      ys[j + 1] = ys[j];
      --j;
    }
    xs[j + 1] = kx;
    ys[j + 1] = ky;
  }
  const int tl = ys[0] < ys[1] ? 0 : 1, bl = 1 - tl;
  const int tr = ys[2] < ys[3] ? 2 : 3, br = 5 - tr;
  const float ox[4] = {xs[tl], xs[tr], xs[br], xs[bl]};
  const float oy[4] = {ys[tl], ys[tr], ys[br], ys[bl]};
  for (int k = 0; k < 4; ++k) {
    xs[k] = ox[k];
    ys[k] = oy[k];
  }
}

}  // namespace

bool TextDet::load(const std::string &plan_path, int max_side,
                   const hp::core::detection::DbParams &db) {
  max_side_ = max_side;
  db_ = db;
  engine_ = std::make_unique<hp::core::engine::TrtEngine>(plan_path);
  if (!engine_->load()) return false;
  const size_t max_px = static_cast<size_t>(max_side + 32) * (max_side + 32);
  d_in_ = CudaPtr<float>(3 * max_px);
  d_pred_ = CudaPtr<float>(max_px);
  d_bitmap_ = CudaPtr<uint8_t>(max_px);
  d_labels_ = CudaPtr<int>(max_px);
  d_compact_ids_ = CudaPtr<int>(max_px);
  d_id_counter_ = CudaPtr<int>(1);
  d_num_boxes_ = CudaPtr<int>(1);
  d_bboxes_ = CudaPtr<GpuDetBox>(kMaxGpuComponents * 2);
  d_jfa_labels_ = CudaPtr<uint32_t>(max_px);
  d_jfa_seeds_ = CudaPtr<uint32_t>(max_px);
  d_jfa_seeds_alt_ = CudaPtr<uint32_t>(max_px);
  d_expand_ = CudaPtr<float>(kMaxGpuComponents);
  d_perim_ = CudaPtr<int>(kMaxGpuComponents);
  d_moments_ = CudaPtr<unsigned long long>(kMaxGpuComponents * 6);
  d_orient_ = CudaPtr<float>(kMaxGpuComponents * 6);
  h_count_ = CudaHostPtr<int>(1);
  h_exp_boxes_ = CudaHostPtr<GpuDetBox>(kMaxGpuComponents);
  engine_->bind_io(d_in_.get(), d_pred_.get());
  return true;
}

ResizeOut TextDet::resize_spec(int page_w, int page_h) {
  page_w_ = page_w;
  page_h_ = page_h;
  ResizeOut o = det_spec(page_w, page_h, max_side_, d_in_.get());
  in_w_ = o.out_w;
  in_h_ = o.out_h;
  return o;
}

float *TextDet::canvas_input(int w, int h) {
  page_w_ = in_w_ = w;
  page_h_ = in_h_ = h;
  return d_in_.get();
}

float TextDet::page_scale(int page_w, int page_h) const {
  return static_cast<float>(std::min(1.0, static_cast<double>(max_side_) / std::max(page_w, page_h)));
}

void TextDet::enqueue(cudaStream_t stream) {
  namespace k = hp::core::kernels;
  nvinfer1::Dims d{};
  d.nbDims = 4;
  d.d[0] = 1;
  d.d[1] = 3;
  d.d[2] = in_h_;
  d.d[3] = in_w_;
  if (!engine_->infer_dynamic(d, stream)) throw std::runtime_error("det execute failed");
  const int w = in_w_, h = in_h_;
  run_w_ = w;
  run_h_ = h;
  run_page_w_ = page_w_;
  run_page_h_ = page_h_;
  k::cuda_threshold_to_u8(d_pred_.get(), d_bitmap_.get(), w, h, db_.thresh, stream);
  k::cuda_gpu_ccl_detect(d_bitmap_.get(), d_pred_.get(), w, h, db_.box_thresh, d_labels_.get(),
                         d_compact_ids_.get(), d_id_counter_.get(), d_bboxes_.get(),
                         d_num_boxes_.get(), nullptr, nullptr, stream);
  // The component count stays on the device: every kernel below covers all
  // kMaxGpuComponents slots. Slots past the page's count hold the CCL init
  // values (no pixels), so they get expand 0 and no label, and collect() reads
  // only the first min(count, kMaxGpuComponents) slots: the boxes are the same
  // as with the count read back first, without a host round trip mid-det.
  constexpr int slots = kMaxGpuComponents;
  k::cuda_accumulate_crack_perimeter(d_compact_ids_.get(), d_bitmap_.get(), w, h, slots,
                                     d_perim_.get(), stream);
  k::cuda_compute_expand_per_comp(d_bboxes_.get(), d_perim_.get(), slots, db_.unclip_ratio,
                                  kMinExpand, kMaxExpand, db_.box_thresh, d_expand_.get(),
                                  stream);
  k::cuda_jfa_expand_labels(d_bitmap_.get(), d_compact_ids_.get(), d_expand_.get(),
                            d_jfa_labels_.get(), w, h, kMaxExpand, d_jfa_seeds_.get(),
                            d_jfa_seeds_alt_.get(), stream);
  GpuDetBox *exp = d_bboxes_.get() + kMaxGpuComponents;
  k::cuda_jfa_extract_oriented(d_jfa_labels_.get(), w, h, exp, slots, d_moments_.get(),
                               d_orient_.get(), stream);
  CUDA_CHECK(cudaMemcpyAsync(h_count_.get(), d_id_counter_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_exp_boxes_.get(), exp, slots * sizeof(GpuDetBox),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(done_, stream));
}

std::vector<Box> TextDet::collect() {
  CUDA_CHECK(cudaEventSynchronize(done_));
  const int w = run_w_, h = run_h_, page_w = run_page_w_, page_h = run_page_h_;
  const int slots = std::min(*h_count_.get(), kMaxGpuComponents);
  std::vector<Box> boxes;
  // Det input is the whole page resized (no pad), so content == input size.
  const float rw = static_cast<float>(w) / page_w, rh = static_cast<float>(h) / page_h;
  for (int i = 0; i < slots; ++i) {
    const GpuDetBox &eb = h_exp_boxes_.get()[i];
    if (eb.pixel_count < 9) continue;
    if (eb.xmax - eb.xmin + 1 < kMinUnclippedSide || eb.ymax - eb.ymin + 1 < kMinUnclippedSide)
      continue;
    float px[4], py[4];
    for (int q = 0; q < 4; ++q) {
      px[q] = eb.ox[q];
      py[q] = eb.oy[q];
    }
    order_quad(px, py);
    Box b;
    for (int q = 0; q < 4; ++q)
      b[q] = {std::clamp(static_cast<int>(std::lround(px[q] / rw)), 0, page_w - 1),
              std::clamp(static_cast<int>(std::lround(py[q] / rh)), 0, page_h - 1)};
    auto len = [](const std::array<int, 2> &a, const std::array<int, 2> &c) {
      return static_cast<int>(std::sqrt(double((a[0] - c[0]) * (a[0] - c[0]) +
                                               (a[1] - c[1]) * (a[1] - c[1]))));
    };
    if (len(b[0], b[1]) <= 3 || len(b[0], b[3]) <= 3) continue;
    boxes.push_back(b);
  }
  return boxes;
}

}  // namespace hp
