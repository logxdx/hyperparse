# Third-party notices

| Component | Where | License |
|---|---|---|
| [liteparse](https://github.com/run-llama/liteparse) | Text-layer routing, glyph recovery and the AGL table, ported to C++ in `src/text_layer.cpp`, `src/glyph_decoder.cpp`, `src/agl_table.h`; LayoutNet's text units, features and table decoding (`table_model/`), ported in `src/tablenet/` | Apache-2.0 |
| [nlohmann/json](https://github.com/nlohmann/json) | fetched into `build/_deps` | MIT |
| [PDFium](https://pdfium.googlesource.com/pdfium/) (run-llama fork) | fetched into `build/_deps` | BSD-3-Clause, installed to `dist/share/licenses/pdfium` |
| [Drogon](https://github.com/drogonframework/drogon), [jsoncpp](https://github.com/open-source-parsers/jsoncpp) | built into `build/_deps` by `tools/build_drogon.sh` | MIT |
| [TensorRT](https://developer.nvidia.com/tensorrt) | linked; libraries copied into `dist/lib` | NVIDIA TensorRT license |

The models (D-FINE layout from SoMark OXR-1.0, PP-OCRv6 and PP-FormulaNet-S
from PaddleOCR, LayoutNet from LlamaIndex's OCRia-nano-1) are not in this repository.
They keep their own licenses, listed with them on Hugging Face. OCRia-nano-1's license
(based on AI Pubs Open RAIL-M) limits commercial use; see its model card.
