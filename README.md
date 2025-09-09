# MambaLite-Micro
A fully C-based, runtime-free inference engine that brings the Mamba sequence model to resource-constrained microcontrollers (MCUs). Unlike the original Mamba implementation, which depends on GPU-specific kernels and lacks ONNX export, MambaLite-Micro enables direct deployment on embedded devices without vendor-specific runtimes.

Key highlights:
- 🚀 **Runtime-free design**: weights exported as plain C arrays, compiled directly into MCU firmware.  
- 💾 **83% memory reduction** through operator fusion and lifetime-aware memory layout.  
- ✅ **100% accuracy consistency** with PyTorch baselines across keyword spotting (KWS) and human activity recognition (HAR).  
- 🔧 **Cross-platform portability**, validated on ESP32S3 and STM32H7 MCUs.  

## Getting Started

### Requirements

```
torch==2.6.0+cu126
torchaudio==2.6.0+cu126
numpy>=1.23
```

### Training

A sample training script is given in `Python/train.py`, please note that only mamba_simple is supported for now.

### Export Weights

The weight export script is given in `Python/export_weights.py`. This generates a C header with weights as `float` arrays for MambaLite-Micro. 

### Build on MCU

- For **ESP32 (ESP-IDF / PlatformIO)**: see `examples/mambakws-esp32-3/`
- For **Arduino (STM32 / other MCUs)**: see `examples/mambahar-arduino-6/`
- For **Generic MCU targets**: see `examples/mambakws-any-10/`  

Each example project includes:

- `include/` → demo wrapper + `sample_input.h`
- `src/` → platform main program

## License

MIT License.

------
## Paper
[MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs](https://arxiv.org/abs/2509.05488)  
Hongjun Xu, Junxi Xia, Weisi Yang, Yueyuan Sui, Stephen Xia  
arXiv preprint arXiv:2509.05488, 2025
## Citation

If you find this project useful in your research, please cite this:

```bibtex
@article{MambaLite-Micro2025,
  title   = {MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs},
  author  = {Hongjun Xu, Junxi Xia, Weisi Yang, Yueyuan Sui, Stephen Xia},
  journal = {arXiv preprint arXiv:2509.05488},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.05488}
}
```

## Directory Structure
```
MambaLite-Micro/
├── csrc/ # Core C implementation (mamba.c, matrix.c, tensor.c, etc.)
├── examples/ # Platform-specific demos
│ ├── mambahar-arduino-6/
│ ├── mambakws-any-10/
│ └── mambakws-esp32-3/
├── Python/ # Training and export scripts
│ ├── train.py
│ └── export_weights.py
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md
```
