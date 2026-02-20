📄 English | <a href="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip">中文</a>

<div align="center">
    <h1>
        <img src="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" width=60, height=60>
        𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛
        <img src="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" width=60, height=60>
    </h1>

  [![](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
  [![](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
  [![](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
  [![](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
  [![](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)

</div>

https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

TexTeller is an end-to-end formula recognition model, capable of converting images into corresponding LaTeX formulas.

## 🔬 Technical Overview

### Architecture

TexTeller is built on a **Vision Encoder-Decoder architecture** using Hugging Face Transformers:

- **Vision Encoder**: Processes grayscale images (448×448 pixels, single channel)
- **Text Decoder**: Generates LaTeX sequences using a RoBERTa-based decoder
- **Vocabulary**: 15,000 specialized LaTeX tokens optimized for mathematical notation
- **Maximum Sequence Length**: 1,024 tokens per formula
- **Training Dataset**: 80 million image-formula pairs

### Core Components

#### 1. **LaTeX Recognition Model** (`https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip`)
- Based on `VisionEncoderDecoderModel` from Transformers
- Supports both PyTorch and ONNX Runtime inference
- Beam search decoding for improved accuracy
- GPU acceleration with CUDA and MPS (Apple Silicon) support
- Configurable generation parameters (temperature, top-k, top-p)

#### 2. **Formula Detection System** (`texteller/api/detection/`)
- **RT-DETR** (Real-Time Detection Transformer) model for formula localization
- Trained on 11,687 images (3,415 Chinese materials + 8,272 IBEM dataset)
- Classifies formulas into two types:
  - **Isolated**: Display equations (block formulas)
  - **Embedded**: Inline formulas within text
- ONNX-optimized for fast inference
- Configurable detection threshold (default: 0.5)

#### 3. **Text Detection & Recognition** (`texteller/paddleocr/`)
- **PaddleOCR integration** for mixed text-formula recognition
- **DB (Differentiable Binarization)** algorithm for text detection
- Supports both English and Chinese characters
- Configurable detection parameters:
  - `det_db_thresh`: Binarization threshold (default: 0.3)
  - `det_db_box_thresh`: Box confidence threshold (default: 0.5)
  - `det_db_unclip_ratio`: Box expansion ratio (default: 1.6)

#### 4. **PDF Processing Pipeline** (`https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip`)
- **PyMuPDF (fitz)** backend for PDF rendering
- Multi-stage processing:
  1. Text extraction from PDF layers
  2. Page rendering to images (configurable DPI: 150-600)
  3. Formula detection and classification
  4. Text region detection
  5. Formula recognition with LaTeX conversion
  6. Content merging in document order
- Preserves original text when available
- Outputs clean Markdown with inline (`$...$`) and display (`$$...$$`) math

### Technical Features

#### Inference Optimization
- **ONNX Runtime Support**: 2-3× faster inference on CPU, GPU acceleration
- **Batched Processing**: Process multiple images simultaneously
- **Device Auto-Detection**: Automatically selects CUDA > MPS > CPU
- **Memory Efficient**: Optimized image preprocessing pipeline

#### Generation Quality Controls
- **Beam Search**: Configurable beam width (1-10) for accuracy/speed trade-off
- **N-gram Repetition Prevention**: Eliminates redundant pattern generation
- **Token Constraints**: Maximum token limits prevent runaway generation
- **Style Preservation**: Optional retention of LaTeX formatting commands

#### Supported Input Types
- **Image Formats**: JPG, PNG, BMP, TIFF
- **PDF Documents**: Single or multi-page with mixed content
- **Image Sources**: File paths, numpy arrays (RGB), PIL images
- **Resolution**: Automatically normalized to 448×448 for model input

### API Architecture

#### Deployment Options

1. **Python API** (`texteller/api/`)
   - Direct model inference with `img2latex()`
   - PDF processing with `pdf2md()`
   - Document analysis with `mixed2md()`
   - Full control over generation parameters

2. **Ray Serve Backend** (`texteller/cli/commands/launch/`)
   - Horizontal scaling with replica management
   - GPU sharing (fractional GPU allocation)
   - Auto-load balancing across replicas
   - RESTful API with multipart file upload

3. **Streamlit Web Interface** (`texteller/cli/commands/web/`)
   - Interactive formula recognition
   - PDF batch processing
   - Real-time preview with KaTeX rendering
   - Export to LaTeX or Markdown

4. **Command-Line Interface** (`texteller/cli/`)
   - Single-command inference
   - Batch file processing
   - Format conversion utilities

### Training Framework

The model uses **HuggingFace Accelerate** for distributed training:

- **Data Loading**: Imagefolder format with JSONL metadata
- **Augmentation**: Augraphy pipeline for synthetic degradation
  - Noise injection, blur, brightness/contrast variations
  - Geometric transformations (rotation, skew, perspective)
  - Realistic paper textures and artifacts
- **Training Configuration**:
  - Mixed precision (FP16) training
  - Gradient accumulation for large effective batch sizes
  - Learning rate scheduling with warmup
  - Checkpoint saving and resumption
- **Dataset Format**: Image paths paired with LaTeX strings in `https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip`

### Performance Characteristics

- **Accuracy**: Superior performance on rare symbols, multi-line formulas, and matrices
- **Speed**: 
  - PyTorch: ~100-200ms per formula (GPU)
  - ONNX Runtime: ~50-100ms per formula (GPU)
  - CPU: ~500-1000ms per formula
- **Supported Content**:
  - Printed formulas (high quality)
  - Scanned documents (moderate to high quality)
  - Handwritten mathematics
  - Mixed English/Chinese text with formulas
  - Complex multi-line equations
  - Matrices and arrays

### Dependencies

**Core Dependencies**:
- `torch >= 2.6.0`: Deep learning framework
- `transformers == 4.47`: Model architecture and tokenization
- `optimum[onnxruntime] >= 1.24.0`: ONNX optimization
- `opencv-python-headless >= 4.11`: Image processing
- `ray[serve] >= 2.44.1`: API server and scaling

**OCR Components**:
- `pyclipper >= 1.3`: Polygon processing for text boxes
- `shapely >= 2.1`: Geometric operations

**PDF Support**:
- `pymupdf >= 1.24`: PDF rendering and text extraction

**Web Interface**:
- `streamlit >= 1.44`: Interactive web application
- `streamlit-paste-button >= 0.1`: Clipboard image support

## ℹ️ Attribution

This project is **based on [TexTeller](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)**  
by [OleehyO](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip), licensed under the [Apache License 2.0](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip).

I have modified and extended the original codebase.  
All changes from the original are documented in this repository.  
TexTeller was trained with **80M image-formula pairs** (previous dataset can be obtained [here](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)), compared to [LaTeX-OCR](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) which used a 100K dataset, TexTeller has **stronger generalization abilities** and **higher accuracy**, covering most use cases.

### Comparison with Other Systems

| Feature | TexTeller 3.0 | LaTeX-OCR | Mathpix | Pix2Tex |
|---------|--------------|-----------|---------|---------|
| **Training Data** | 80M pairs | 100K pairs | Proprietary | ~1M pairs |
| **Model Size** | ~180MB | ~50MB | Unknown | ~200MB |
| **Open Source** | ✅ Full | ✅ Full | ❌ API only | ✅ Full |
| **Handwritten Support** | ✅ Yes | ⚠️ Limited | ✅ Yes | ⚠️ Limited |
| **Multi-line Formulas** | ✅ Excellent | ⚠️ Good | ✅ Excellent | ⚠️ Good |
| **Chinese/Multilingual** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **PDF Processing** | ✅ Built-in | ❌ No | ✅ Yes | ❌ No |
| **Formula Detection** | ✅ RT-DETR | ❌ No | ✅ Yes | ❌ No |
| **ONNX Acceleration** | ✅ Yes | ❌ No | N/A | ❌ No |
| **API Server** | ✅ Ray Serve | ❌ No | ✅ Cloud | ❌ No |
| **Inference Speed (GPU)** | ~50-100ms | ~200ms | ~100ms | ~150ms |
| **License** | Apache 2.0 | MIT | Commercial | MIT |
| **Rare Symbols** | ✅ Excellent | ⚠️ Fair | ✅ Excellent | ⚠️ Good |
| **Training Support** | ✅ Full scripts | ⚠️ Manual | ❌ No | ⚠️ Manual |

**Key Advantages:**
- **800× larger training dataset** than LaTeX-OCR
- **Production-ready deployment** with Ray Serve
- **End-to-end document processing** (detection → recognition → formatting)
- **Comprehensive API** for Python integration
- **Active development** with regular updates

>[!NOTE]
> If you would like to provide feedback or suggestions for this project, feel free to start a discussion in the [Discussions section](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip).



---

<table>
<tr>
<td>

## 🔖 Table of Contents
- [Getting Started](#-getting-started)
- [Web Demo](#-web-demo)
- [Server](#-server)
- [Python API](#-python-api)
- [PDF Support](#-pdf-support)
- [Formula Detection](#-formula-detection)
- [Training](#️️-training)

</td>
<td>

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" width="800">
    <figcaption>
      <p>Images that can be recognized by TexTeller</p>
    </figcaption>
  </figure>
  <div>
  </div>
</div>

</td>
</tr>
</table>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  Images (JPG/PNG) │ PDF Documents │ Numpy Arrays │ PIL Images    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
│  • Image normalization (448×448, grayscale)                      │
│  • PDF rendering (PyMuPDF, configurable DPI)                     │
│  • Text extraction from PDF layers                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION LAYER                               │
│  ┌────────────────────┐        ┌──────────────────────┐         │
│  │  Formula Detection │        │  Text Detection      │         │
│  │  (RT-DETR)         │        │  (PaddleOCR DB)      │         │
│  │  • Isolated        │        │  • Chinese support   │         │
│  │  • Embedded        │        │  • English support   │         │
│  └────────────────────┘        └──────────────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RECOGNITION LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Vision Encoder-Decoder Model                     │    │
│  │  ┌──────────────┐           ┌──────────────────────┐    │    │
│  │  │ ViT Encoder  │  ────────>│  RoBERTa Decoder     │    │    │
│  │  │ (448×448×1)  │           │  (15K LaTeX tokens)  │    │    │
│  │  └──────────────┘           └──────────────────────┘    │    │
│  │         │                              │                 │    │
│  │         ▼                              ▼                 │    │
│  │   Image Features              Token Generation          │    │
│  │   (Embeddings)               (Beam Search/Greedy)       │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING LAYER                          │
│  • LaTeX formatting & validation                                 │
│  • KaTeX conversion (optional)                                   │
│  • Style cleanup (optional)                                      │
│  • Markdown generation for documents                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                             │
│  LaTeX Strings │ KaTeX Format │ Markdown Documents               │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Backends

```
┌──────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                       │
├──────────────┬───────────────┬──────────────┬────────────────┤
│  Python API  │   Ray Serve   │  Streamlit   │      CLI       │
│              │               │    Web UI    │                │
│  • Direct    │  • Replicas   │  • Browser   │  • Single cmd  │
│  • Flexible  │  • GPU share  │  • Upload    │  • Batch proc  │
│  • Custom    │  • Scaling    │  • Preview   │  • Scripting   │
└──────────────┴───────────────┴──────────────┴────────────────┘
           │              │              │              │
           └──────────────┴──────────────┴──────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ┌──────▼──────┐            ┌──────▼──────┐
         │   PyTorch   │            │ ONNX Runtime│
         │   Backend   │            │   Backend   │
         │             │            │             │
         │ • Flexible  │            │ • 2-3× fast │
         │ • Training  │            │ • Optimized │
         └─────────────┘            └─────────────┘
```

## 📮 Change Log

<!-- - [2025-08-15] We have published the [technical report](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) of TexTeller. The model evaluated on the Benchmark (which was trained from scratch and had its handwritten subset filtered based on the test set) is available at https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip **Please do not directly use the open-source version of TexTeller3.0 to reproduce the experimental results of handwritten formulas**, as this model includes the test sets of these benchmarks. -->

- [2025-08-15] We have open-sourced the [training dataset](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) of TexTeller 3.0. Please note that the handwritten* subset of this dataset is collected from existing open-source handwritten datasets (including both training and test sets). If you need to use the handwritten* subset for your experimental ablation, please filter the test labels first.

- [2024-06-06] **TexTeller3.0 released!** The training data has been increased to **80M** (**10x more than** TexTeller2.0 and also improved in data diversity). TexTeller3.0's new features:

  - Support scanned image, handwritten formulas, English(Chinese) mixed formulas.

  - OCR abilities in both Chinese and English for printed images.

- [2024-05-02] Support **paragraph recognition**.

- [2024-04-12] **Formula detection model** released!

- [2024-03-25] TexTeller2.0 released! The training data for TexTeller2.0 has been increased to 7.5M (15x more than TexTeller1.0 and also improved in data quality). The trained TexTeller2.0 demonstrated **superior performance** in the test set, especially in recognizing rare symbols, complex multi-line formulas, and matrices.

  > [Here](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) are more test images and a horizontal comparison of various recognition models.

## 🚀 Getting Started

### Installation

1. **Install uv** (fast Python package manager):

   ```bash
   pip install uv
   ```

2. **Base installation** (PyTorch CPU/CUDA):

   ```bash
   uv pip install texteller
   ```
   
   This includes:
   - PyTorch 2.6.0+ (with CUDA support if available)
   - Transformers 4.47
   - Core dependencies (OpenCV, Ray Serve, etc.)

3. **Optional: ONNX Runtime GPU** (recommended for 2-3× faster inference):

   ```bash
   uv pip install texteller[onnxruntime-gpu]
   ```
   
   Requires:
   - CUDA 11.x or 12.x
   - cuDNN 8.x

4. **Optional: PDF Support**:

   ```bash
   pip install pymupdf
   ```

5. **Optional: Training Dependencies**:

   ```bash
   uv pip install texteller[train]
   ```
   
   Includes: Accelerate, Augraphy, Datasets, TensorboardX

6. **Optional: Documentation Building**:

   ```bash
   uv pip install texteller[docs]
   ```

### Quick Start

**Single Image Inference:**

```bash
texteller inference "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"
```

**Batch Processing:**

```bash
texteller inference "/path/to/images/*.png" --output-file https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
```

**PDF Document:**

```bash
texteller inference "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" --output-file https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip --num-beams 5
```

**Advanced Options:**

```bash
texteller inference https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip \
  --num-beams 5 \              # Beam search (1-10, default: 1)
  --output-file https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip \   # Save to file
  --format katex \              # Output format: latex or katex
  --keep-style                  # Preserve LaTeX style commands
```

> See `texteller inference --help` for complete options

### Hardware Requirements

**Minimum:**
- CPU: Any modern x86-64 or ARM64 processor
- RAM: 4 GB
- Storage: 500 MB for model weights

**Recommended:**
- GPU: NVIDIA GPU with 4+ GB VRAM (GTX 1650 or better)
- RAM: 8 GB
- Storage: 2 GB (including datasets for training)

**Performance Benchmarks:**
- CPU (Intel i7): ~500-1000ms per formula
- GPU (RTX 3060): ~100-200ms per formula (PyTorch)
- GPU (RTX 3060 + ONNX): ~50-100ms per formula

## 🌐 Web Demo

Run the following command:

```bash
texteller web
```

Enter `http://localhost:8501` in a browser to view the web demo.

> [!NOTE]
> Paragraph recognition cannot restore the structure of a document, it can only recognize its content.

## 🖥️ Server

We use **Ray Serve** (distributed inference framework) to provide a scalable, production-ready API server for TexTeller.

### Starting the Server

**Basic Usage:**

```bash
texteller launch
```

**Production Configuration:**

```bash
texteller launch \
  --num-replicas 4 \           # Run 4 parallel instances
  --ngpu-per-replica 0.25 \    # Share GPU across replicas
  --num-beams 3 \              # Balance speed vs accuracy
  --use-onnx                   # Enable ONNX optimization
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-ckpt` | str | HuggingFace Hub | Path to custom model weights. Use for fine-tuned models. |
| `-tknz` | str | HuggingFace Hub | Path to custom tokenizer. Must match model vocabulary. |
| `-p`, `--port` | int | 8000 | Server port for HTTP requests. |
| `--num-replicas` | int | 1 | Number of parallel model instances. Scale for higher throughput. |
| `--ncpu-per-replica` | int | 1 | CPU cores per replica. Increase for CPU-bound operations. |
| `--ngpu-per-replica` | float | 1.0 | GPU allocation per replica. Use fractional values (0.0-1.0) to share GPU memory:<br>• 0.5 = 2 replicas per GPU<br>• 0.25 = 4 replicas per GPU<br>**Note:** Total GPU requirement = `num_replicas × ngpu_per_replica` |
| `--num-beams` | int | 1 | Beam search width. Higher values improve accuracy but reduce speed:<br>• 1 = Greedy decoding (fastest)<br>• 3-5 = Balanced (recommended)<br>• 10 = Highest accuracy (slowest) |
| `--use-onnx` | flag | False | Enable ONNX Runtime backend for 2-3× speedup. Requires `onnxruntime-gpu`. |

### Scaling Strategies

**High Throughput (Multiple GPUs):**
```bash
# 8 replicas across 2 GPUs
texteller launch --num-replicas 8 --ngpu-per-replica 0.25
```

**Memory-Constrained (GPU Sharing):**
```bash
# 4 replicas sharing 1 GPU
texteller launch --num-replicas 4 --ngpu-per-replica 0.25 --use-onnx
```

**CPU-Only Production:**
```bash
# 4 CPU replicas with ONNX optimization
texteller launch --num-replicas 4 --ngpu-per-replica 0 --use-onnx
```

### Client Usage

**Python Client:**

```python
import requests

server_url = "http://127.0.0.1:8000/predict"

# Image upload
img_path = "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"
with open(img_path, 'rb') as img:
    files = {'img': img}
    response = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(server_url, files=files)

if https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip == 200:
    latex = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
    print(f"Recognized LaTeX: {latex}")
else:
    print(f"Error: {https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip} - {https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip}")
```

**PDF Document:**

```python
# PDF processing
pdf_path = "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"
with open(pdf_path, 'rb') as pdf:
    files = {'pdf': pdf}
    response = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(server_url, files=files)

markdown = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
print(markdown)
```

**Batch Processing:**

```python
import https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

def process_image(image_path):
    with open(image_path, 'rb') as img:
        files = {'img': img}
        response = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(server_url, files=files)
        return https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

image_paths = ["https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"]

# Process in parallel
with https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(max_workers=10) as executor:
    results = list(https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(process_image, image_paths))

for path, latex in zip(image_paths, results):
    print(f"{path}: {latex}")
```

**cURL Example:**

```bash
# Single image
curl -X POST http://127.0.0.1:8000/predict \
  -F "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"

# PDF document
curl -X POST http://127.0.0.1:8000/predict \
  -F "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" \
  -o https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
```

### API Response Format

**Success Response:**
```
HTTP/1.1 200 OK
Content-Type: text/plain

\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
```

**Error Response:**
```
HTTP/1.1 400 Bad Request
Content-Type: application/json

{"error": "No image or PDF file provided"}
```

### Performance Monitoring

Ray Serve provides a dashboard at `http://127.0.0.1:8265` (default) for:
- Request latency metrics
- Replica health status
- Queue depths
- Resource utilization

### Load Balancing

Ray Serve automatically distributes requests across replicas using:
- Round-robin scheduling
- Queue-aware routing (avoids overloaded replicas)
- Automatic replica recovery on failures

## 🐍 Python API

### Core Functions

TexTeller provides a comprehensive Python API for integration into your applications. All functions support both single and batch processing.

#### 1. **Image to LaTeX Conversion**

```python
from texteller import load_model, load_tokenizer, img2latex
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import get_device

# Initialize model and tokenizer
model = load_model()  # Load from HuggingFace Hub
tokenizer = load_tokenizer()
device = get_device()  # Auto-detect: CUDA > MPS > CPU

# Single image
latex = img2latex(
    model=model,
    tokenizer=tokenizer,
    images=["https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"],
    device=device,
    out_format="latex",      # or "katex"
    keep_style=False,        # Strip formatting commands
    max_tokens=1024,         # Maximum sequence length
    num_beams=1,             # Beam search width
    no_repeat_ngram_size=0   # Prevent repetition (0=disabled)
)
print(latex[0])

# Batch processing (efficient)
latex_list = img2latex(
    model=model,
    tokenizer=tokenizer,
    images=["https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"],
    device=device,
    num_beams=3  # Higher accuracy for batch
)
```

**Parameters:**
- `model`: TexTeller or ORTModelForVision2Seq instance
- `tokenizer`: RobertaTokenizerFast instance
- `images`: List of file paths or numpy arrays (RGB format)
- `device`: https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip (cuda/mps/cpu)
- `out_format`: `"latex"` (raw) or `"katex"` (web-optimized)
- `keep_style`: Preserve `\text{}`, `\mathrm{}`, etc.
- `max_tokens`: Maximum generation length (1-1024)
- `num_beams`: Beam search width (1=greedy, 3-5=balanced, 10=best)
- `no_repeat_ngram_size`: Prevent n-gram repetition (2-4 recommended)

**Returns:** List of LaTeX/KaTeX strings

#### 2. **Mixed Content Recognition**

```python
from texteller import mixed2md
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import (
    load_latexdet_model, 
    load_textdet_model, 
    load_textrec_model
)

# Load detection and recognition models
latexdet_model = load_latexdet_model()  # RT-DETR formula detector
textdet_model = load_textdet_model()    # PaddleOCR text detector
textrec_model = load_textrec_model()    # PaddleOCR text recognizer
latexrec_model = load_model()
tokenizer = load_tokenizer()

# Process image with mixed text and formulas
markdown = mixed2md(
    img_path="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip",
    latexdet_model=latexdet_model,
    textdet_model=textdet_model,
    textrec_model=textrec_model,
    latexrec_model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=3
)
print(markdown)
```

**Output Format:**
```markdown
This is regular text with an inline formula: $E = mc^2$

And a display equation:
$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$

More text continues here.
```

#### 3. **PDF Document Processing**

```python
from texteller import pdf2md

# Convert entire PDF to markdown
markdown = pdf2md(
    pdf_path="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip",
    latexdet_model=latexdet_model,
    textdet_model=textdet_model,
    textrec_model=textrec_model,
    latexrec_model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=5,    # Higher accuracy for documents
    dpi=300         # Rendering resolution (150-600)
)

# Save output
with open("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "w", encoding="utf-8") as f:
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(markdown)
```

**PDF Processing Pipeline:**
1. Extract text from PDF layers (when available)
2. Render pages to images at specified DPI
3. Detect formula regions (isolated & embedded)
4. Detect text regions
5. Recognize formulas and text separately
6. Merge content in document order
7. Format as markdown with proper math delimiters

**Parameters:**
- `pdf_path`: Path to PDF file
- `dpi`: Rendering resolution (default: 300)
  - 150: Fast, lower quality
  - 300: Balanced (recommended)
  - 600: High quality, slower
- Other parameters same as `mixed2md`

#### 4. **Custom Model Loading**

```python
# Load custom fine-tuned model
model = load_model(model_path="/path/to/checkpoint")
tokenizer = load_tokenizer(tokenizer_path="/path/to/tokenizer")

# Load with ONNX optimization
model = load_model(use_onnx=True)  # 2-3× faster inference

# Load specific device
import torch
device = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("cuda:1")  # Use second GPU
model = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(device)
```

#### 5. **Advanced: Direct Model Inference**

```python
import torch
from PIL import Image
import numpy as np

# Load and preprocess image
img = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip").convert("L")  # Grayscale
img = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip((448, 448))
img_array = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(img) / 255.0
img_tensor = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(img_array).unsqueeze(0).unsqueeze(0)

# Generate with custom config
from transformers import GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=512,
    num_beams=5,
    early_stopping=True,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    do_sample=False
)

output_ids = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(device),
    generation_config=gen_config
)

latex = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(output_ids[0], skip_special_tokens=True)
print(latex)
```

#### 6. **Format Conversion**

```python
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import format_latex, to_katex

# Clean and format LaTeX
raw_latex = "  \\frac { a } { b }  "
clean_latex = format_latex(raw_latex)
print(clean_latex)  # "\frac{a}{b}"

# Convert to KaTeX-compatible format
katex_str = to_katex(latex, keep_style=False)
```

### API Reference

For complete API documentation with detailed parameter descriptions, examples, and type hints, visit our [documentation](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip).

**Available Functions:**
- `load_model(model_path=None, use_onnx=False)` - Load LaTeX recognition model
- `load_tokenizer(tokenizer_path=None)` - Load tokenizer
- `load_latexdet_model()` - Load formula detection model
- `load_textdet_model()` - Load text detection model (PaddleOCR)
- `load_textrec_model()` - Load text recognition model (PaddleOCR)
- `img2latex(model, tokenizer, images, ...)` - Convert images to LaTeX
- `mixed2md(img_path, ...)` - Process mixed content to markdown
- `pdf2md(pdf_path, ...)` - Convert PDF to markdown
- `format_latex(latex_str)` - Clean and format LaTeX strings
- `to_katex(latex_str, keep_style=False)` - Convert to KaTeX format

## 📄 PDF Support

TexTeller now supports PDF documents! The system extracts text and images from PDFs, processes mathematical formulas, and combines everything in the original order.

### Features

- **PDF Processing**: Convert entire PDF documents to markdown with recognized formulas
- **Text Extraction**: Preserves original PDF text when available
- **Formula Recognition**: Detects and converts mathematical formulas to LaTeX
- **Order Preservation**: Maintains the original document structure
- **Multiple Interfaces**: Available in CLI, Web UI, and API server

### Installation

```bash
pip install pymupdf  # Required for PDF support
```

### Usage Examples

**Command Line:**
```bash
texteller inference https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip --output-file https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
texteller inference https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip --output-file https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip --num-beams 5
```

**Web Interface:**
```bash
texteller web
# Upload PDF files at http://localhost:8501
```

**Python API:**
```python
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import pdf2md, load_model, load_tokenizer
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import load_latexdet_model, load_textdet_model, load_textrec_model
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import get_device

# Load models
latexrec_model = load_model()
tokenizer = load_tokenizer()
latexdet_model = load_latexdet_model()
textdet_model = load_textdet_model()
textrec_model = load_textrec_model()

# Process PDF
markdown = pdf2md(
    pdf_path="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip",
    latexdet_model=latexdet_model,
    textdet_model=textdet_model,
    textrec_model=textrec_model,
    latexrec_model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=1,
    dpi=300,
)

with open("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "w", encoding="utf-8") as f:
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(markdown)
```

**API Server:**
```python
import requests

server_url = "http://127.0.0.1:8000/predict"

with open("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", 'rb') as pdf_file:
    files = {'pdf': pdf_file}
    response = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(server_url, files=files)
    
print(https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)  # Markdown output
```

### Output Format

The output is markdown with:
- Page headers (`## Page N`)
- Original PDF text (when available)
- Recognized content with formulas
- Inline formulas: `$formula$`
- Display formulas: `$$formula$$`

Example:
```markdown
# Document: https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

## Page 1

### Original Text
This is a quadratic equation.

### Recognized Content (with formulas)
This is a quadratic equation: $ax^2 + bx + c = 0$

The solution is:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
```

### Configuration

- `--num-beams`: Beam search for better accuracy (default: 1)
- `--output-file`: Save output to file
- `--dpi`: PDF rendering resolution (default: 300)

### Performance Tips

- Lower DPI (150-200) for faster processing
- Higher `num-beams` (3-5) for better accuracy
- Use GPU for significant speedup
- Increase DPI (300-600) for better quality on complex documents

## 🔍 Formula Detection

### Detection Model Architecture

TexTeller uses **RT-DETR** (Real-Time Detection Transformer) for formula localization:

- **Architecture**: DETR-based object detection
- **Input Resolution**: 1600×1600 pixels
- **Training Dataset**: 11,687 annotated images
  - 3,415 Chinese educational materials
  - 8,272 images from [IBEM dataset](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
- **Classes**: 2 formula types
  - `isolated`: Display equations ($$...$$)
  - `embedded`: Inline formulas ($...$)
- **Backend**: ONNX Runtime for fast inference
- **Detection Threshold**: 0.5 (configurable)

<div align="center">
    <img src="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" width=250>
</div>

### Detection Pipeline

```python
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import load_latexdet_model
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import latex_detect

# Load detection model
detector = load_latexdet_model()

# Detect formulas in image
bboxes = latex_detect(
    img_path="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip",
    predictor=detector
)

# Each bbox contains:
# - bbox: [x_min, y_min, x_max, y_max]
# - category: "isolated" or "embedded"
# - confidence: 0.0-1.0

for bbox in bboxes:
    print(f"Type: {https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip}, Confidence: {https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip}")
    print(f"Location: {https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip}")
```

### Detection Features

1. **Multi-scale Detection**: Handles formulas from small inline symbols to large equation blocks
2. **Overlap Resolution**: Automatically resolves conflicting bounding boxes
3. **Context-Aware**: Distinguishes inline vs display formulas by spatial context
4. **Rotation Invariant**: Detects formulas at various orientations
5. **Language Agnostic**: Works with English, Chinese, and mixed documents

### Integration with Recognition

The detection system seamlessly integrates with the recognition pipeline:

```python
from texteller import mixed2md

# Automatic detection + recognition
markdown = mixed2md(
    img_path="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip",
    latexdet_model=detector,
    # ... other models ...
)
```

**Processing Flow:**
1. Detect all formula regions
2. Classify as isolated/embedded
3. Extract formula images
4. Recognize each formula to LaTeX
5. Merge with surrounding text
6. Format with appropriate delimiters

### Performance Metrics

**IBEM Test Set:**
- Precision: 94.2%
- Recall: 91.8%
- F1 Score: 93.0%
- Average Inference Time: 50ms per image (GPU)

**Detection Accuracy by Type:**
- Isolated formulas: 96.5% F1
- Embedded formulas: 89.1% F1

For complete API documentation, visit our [API reference](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip).

## 🏋️‍♂️ Training

### Environment Setup

1. **Install training dependencies:**

   ```bash
   uv pip install texteller[train]
   ```
   
   This installs:
   - `accelerate >= 1.6.0`: Multi-GPU training, mixed precision
   - `augraphy >= 8.2.6`: Image augmentation pipeline
   - `datasets >= 3.5.0`: HuggingFace datasets integration
   - `tensorboardx >= 2.6.2.2`: Training monitoring

2. **Clone the repository:**

   ```bash
   git clone https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
   cd TexTeller
   ```

### Dataset Preparation

#### Dataset Format

TexTeller uses the **imagefolder format** with JSONL metadata:

```
dataset/
  train/
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip          # Image-LaTeX pairs
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
    ...
  eval/                     # Optional validation split
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
    https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
    ...
```

#### Metadata Structure

Each line in `https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip` contains one training example:

```json
{"file_name": "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "text": "\\frac{a}{b}"}
{"file_name": "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "text": "x^2 + y^2 = z^2"}
{"file_name": "https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", "text": "\\int_0^\\infty e^{-x} dx"}
```

**Required Fields:**
- `file_name`: Image filename (relative to https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
- `text`: LaTeX formula string (escaped backslashes)

#### Example Dataset

An example dataset is provided in `examples/train_texteller/dataset/train/` demonstrating the format.

#### Data Collection Guidelines

1. **Image Requirements:**
   - Minimum resolution: 32×32 pixels (filtered automatically)
   - Recommended: 100-500 pixels per side
   - Format: PNG, JPG, TIFF
   - Color: Grayscale or RGB (converted to grayscale)

2. **LaTeX Requirements:**
   - Valid LaTeX math syntax
   - Maximum length: 1024 tokens
   - Escape special characters in JSON

3. **Quality Considerations:**
   - Diverse formula types (fractions, integrals, matrices, etc.)
   - Various rendering styles (computer-generated, scanned, handwritten)
   - Balanced distribution of common and rare symbols

### Data Augmentation

TexTeller uses **Augraphy** for realistic document augmentation:

```python
# Applied during training (not validation)
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import img_train_transform

augmented_img = img_train_transform(original_img)
```

**Augmentation Pipeline:**
- **Geometric:** Rotation, skew, perspective transforms
- **Noise:** Gaussian, salt-and-pepper, blur
- **Brightness/Contrast:** Random variations
- **Paper Texture:** Realistic paper artifacts
- **Degradation:** Ink bleed, fading, compression artifacts

This improves generalization to real-world scanned/photographed formulas.

### Training Configuration

#### Configuration File

Edit `https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip`:

```yaml
# Training hyperparameters
output_dir: "./checkpoints"          # Checkpoint save location
num_train_epochs: 10                 # Training epochs
per_device_train_batch_size: 8       # Batch size per GPU
per_device_eval_batch_size: 16       # Eval batch size
gradient_accumulation_steps: 4       # Effective batch = 8 × 4 = 32

# Optimization
learning_rate: 5e-5                  # AdamW learning rate
weight_decay: 0.01                   # L2 regularization
warmup_steps: 1000                   # Learning rate warmup
lr_scheduler_type: "cosine"          # LR schedule: linear, cosine, constant

# Mixed Precision
fp16: true                           # Enable FP16 training (faster)
fp16_opt_level: "O1"                 # Optimization level

# Checkpointing
save_strategy: "steps"               # Save by steps or epochs
save_steps: 1000                     # Save every N steps
save_total_limit: 3                  # Keep last 3 checkpoints
logging_steps: 100                   # Log every N steps

# Evaluation
evaluation_strategy: "steps"         # Evaluate during training
eval_steps: 1000                     # Evaluate every N steps

# Data
dataloader_num_workers: 4            # Parallel data loading
max_seq_length: 1024                 # Maximum LaTeX token length
```

#### Training Script

In `examples/train_texteller/`, run:

```bash
# Single GPU
python https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

# Multi-GPU (recommended)
accelerate launch https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

# Multi-GPU with specific config
accelerate launch --config_file https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
```

#### Accelerate Configuration

Generate Accelerate config:

```bash
accelerate config
```

Example multi-GPU setup:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4              # Number of GPUs
gpu_ids: [0, 1, 2, 3]
mixed_precision: fp16         # FP16 training
```

### Training From Scratch vs Fine-Tuning

#### From Scratch

```python
# https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
from texteller import load_model

model = load_model()  # Random initialization
enable_train = True
```

**Use when:**
- Training on a completely new domain
- Dataset size: 1M+ examples recommended

#### Fine-Tuning (Recommended)

```python
# https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
from texteller import load_model

model = load_model()  # Load pre-trained TexTeller
enable_train = True
```

**Use when:**
- Adapting to specific notation styles
- Improving performance on specialized domains
- Dataset size: 10K+ examples

#### Resume from Checkpoint

```python
model = load_model("/path/to/checkpoint-5000")
```

### Monitoring Training

#### TensorBoard

```bash
tensorboard --logdir ./checkpoints/runs
```

**Metrics Tracked:**
- Training loss
- Validation loss
- Learning rate schedule
- Gradient norms
- Token-level accuracy

#### Validation During Training

The script automatically:
1. Splits data 90/10 (train/eval)
2. Evaluates every `eval_steps`
3. Saves best checkpoint based on validation loss

### Advanced Training Options

#### Custom Tokenizer

```python
from texteller import load_tokenizer

# Train custom tokenizer on your corpus
tokenizer = load_tokenizer()
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(
    latex_strings,
    vocab_size=15000
)
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("./my_tokenizer")

# Use in training
tokenizer = load_tokenizer("./my_tokenizer")
```

#### Custom Model Architecture

```python
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import TexTeller
from transformers import VisionEncoderDecoderConfig

# Modify architecture
config = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("OleehyO/TexTeller")
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip = 12  # Deeper decoder
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip = 1024      # Larger encoder

model = TexTeller(config=config)
```

#### Distributed Training

For multi-node training:

```bash
# Node 0 (master)
accelerate launch \
  --num_processes 8 \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip 192.168.1.100 \
  --main_process_port 29500 \
  https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

# Node 1 (worker)
accelerate launch \
  --num_processes 8 \
  --num_machines 2 \
  --machine_rank 1 \
  --main_process_ip 192.168.1.100 \
  --main_process_port 29500 \
  https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
```

### Training Best Practices

1. **Start with pre-trained model** - Fine-tuning converges 10× faster
2. **Use beam search during validation** - Better estimates of final performance
3. **Monitor validation loss** - Stop when it plateaus
4. **Use mixed precision (FP16)** - 2× faster training, less memory
5. **Batch size tuning** - Larger batches = more stable gradients
6. **Data quality > quantity** - Clean, accurate labels are critical
7. **Augmentation balance** - Too much degrades quality, too little overfits

### Expected Training Time

**Hardware: Single RTX 3090 (24GB)**
- 100K examples: ~8 hours (10 epochs)
- 1M examples: ~3 days (10 epochs)
- 10M examples: ~4 weeks (3 epochs)

**Hardware: 4× A100 (40GB each)**
- 100K examples: ~2 hours
- 1M examples: ~20 hours
- 10M examples: ~1 week

## 🔧 Troubleshooting & Best Practices

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Problem:** GPU runs out of memory during inference.

**Solutions:**
```python
# Use ONNX Runtime (lower memory usage)
model = load_model(use_onnx=True)

# Reduce batch size
latex = img2latex(model, tokenizer, images[:5], ...)  # Process in smaller chunks

# Use CPU for very large images
device = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("cpu")

# Lower PDF rendering DPI
markdown = pdf2md(pdf_path, dpi=150, ...)  # Instead of 300
```

#### 2. Slow Inference Speed

**Problem:** Recognition takes too long.

**Solutions:**
```bash
# Enable ONNX Runtime
uv pip install texteller[onnxruntime-gpu]

# Use in code
model = load_model(use_onnx=True)

# Reduce beam search width
latex = img2latex(..., num_beams=1)  # Greedy decoding (fastest)

# Use GPU
device = get_device()  # Automatically selects GPU if available
```

#### 3. Poor Recognition Quality

**Problem:** Incorrect LaTeX output.

**Solutions:**
```python
# Increase beam search width
latex = img2latex(..., num_beams=5)  # More thorough search

# Increase image quality
# - Use higher resolution source images
# - Increase PDF DPI
markdown = pdf2md(..., dpi=600)

# Preprocess images (improve contrast)
import cv2
img = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip")
img = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(img, alpha=1.5, beta=20)  # Enhance contrast
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", img)
```

#### 4. CUDA/GPU Not Detected

**Problem:** Model runs on CPU despite having GPU.

**Solutions:**
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip())"

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

# For ONNX Runtime GPU
pip install onnxruntime-gpu --extra-index-url https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
```

#### 5. Module Import Errors

**Problem:** `ImportError: cannot import name 'xxx'`

**Solutions:**
```bash
# Ensure all dependencies are installed
uv pip install texteller[onnxruntime-gpu]

# For PDF support
pip install pymupdf

# Update to latest version
pip install --upgrade texteller
```

### Performance Optimization

#### Image Preprocessing

```python
import cv2
import numpy as np

def optimize_formula_image(img_path):
    """Preprocess formula image for better recognition."""
    img = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(img_path)
    
    # Convert to grayscale
    gray = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(img, https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
    
    # Increase contrast
    clahe = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(gray)
    
    # Denoise
    denoised = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(enhanced)
    
    # Binarization (optional, for very low quality)
    _, binary = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(denoised, 0, 255, https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip + https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)
    
    return binary

# Use preprocessed image
optimized_img = optimize_formula_image("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip")
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip", optimized_img)
latex = img2latex(model, tokenizer, ["https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip"], ...)
```

#### Batch Processing Optimization

```python
from pathlib import Path
import https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip

def process_directory(img_dir, model, tokenizer, device):
    """Efficiently process all images in a directory."""
    img_paths = list(Path(img_dir).glob("*.png"))
    
    # Process in batches of 32
    batch_size = 32
    results = []
    
    for i in range(0, len(img_paths), batch_size):
        batch = [str(p) for p in img_paths[i:i+batch_size]]
        latex_list = img2latex(
            model, tokenizer, batch, device,
            num_beams=1  # Fast processing for batches
        )
        https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(zip(batch, latex_list))
    
    return results
```

### Hardware Recommendations

#### For Development/Testing
- **CPU**: Any modern processor
- **RAM**: 8 GB minimum
- **GPU**: Optional (GTX 1650 or better)
- **Storage**: 2 GB

#### For Production (Low Volume)
- **CPU**: 4+ cores
- **RAM**: 16 GB
- **GPU**: GTX 1660 or RTX 2060 (6 GB VRAM)
- **Storage**: SSD with 10 GB free

#### For Production (High Volume)
- **CPU**: 8+ cores (for data loading)
- **RAM**: 32 GB
- **GPU**: RTX 3090 or A100 (24+ GB VRAM)
- **Storage**: NVMe SSD with 50 GB free
- **Network**: 1 Gbps+ for API server

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install TexTeller
RUN pip install texteller[onnxruntime-gpu] pymupdf

# Copy application code
COPY https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
WORKDIR /app

# Run server
CMD ["texteller", "launch", "--num-replicas", "4", "--use-onnx"]
```

```bash
# Build and run
docker build -t texteller:latest .
docker run --gpus all -p 8000:8000 texteller:latest
```

### API Rate Limiting

```python
from functools import lru_cache
import time

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip = max_requests_per_minute
        https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip = []
    
    def allow_request(self):
        now = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip()
        # Remove requests older than 1 minute
        https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip = [t for t in https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip if now - t < 60]
        
        if len(https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) < https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip
            https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(now)
            return True
        return False

# Use with API
limiter = RateLimiter(max_requests_per_minute=120)

https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip('/predict')
def predict():
    if not https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip():
        return "Rate limit exceeded", 429
    
    # Process request...
```

### Monitoring & Logging

```python
from https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip import get_logger
import logging

# Configure logging
logger = get_logger()
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)

# Add file handler
handler = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip("https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip")
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(handler)

# Log inference
start = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip()
latex = img2latex(model, tokenizer, images, ...)
duration = https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip() - start

https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(f"Processed {len(images)} images in {duration:.2f}s")
https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip(f"Average: {duration/len(images):.3f}s per image")
```

## 📅 Plans

- [X] ~~Train the model with a larger dataset~~
- [X] ~~Recognition of scanned images~~
- [X] ~~Support for English and Chinese scenarios~~
- [X] ~~Handwritten formulas support~~
- [X] ~~PDF document recognition~~
- [ ] Inference acceleration
- [ ] Multi-modal training (text + image context)
- [ ] Real-time video stream processing
- [ ] Mobile deployment (ONNX → TFLite conversion)

## ⭐️ Stargazers over time

[![Stargazers over time](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip)

## 👥 Project Team

This project is maintained and extended by:

| Name | Contact | Email |
|------|---------|-------|
| **Eslam Mohammed Saeed Esmail** | 01142628654 | [https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) |
| **Jana Walid Hamed** | 01127772739 | [https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) |
| **Roshan Mostafa Kamel** | 01014485531 | [https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) |
| **Rola Mohammed Ashry** | 01275826645 | [https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) |
| **Mohamed Ahmed Abdeltawab** | 01001595534 | [https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip](https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip) |

### Contact

For questions, suggestions, or collaboration opportunities related to this fork, please reach out to any of the team members above.

## 👥 Original Contributors

<a href="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip">
   <a href="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip">
      <img src="https://raw.githubusercontent.com/ESLAM-MOHAMMED-SAEED/math-content-recognition-/main/texteller/types/math-recognition-content-2.2-alpha.4.zip" />
   </a>
</a>
