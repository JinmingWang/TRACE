# TRACE: Trajectory Recovery with State Propagation Diffusion for Urban Mobility

This repository contains the implementation of Trace, a novel diffusion-based model for trajectory recovery and generation. The model leverages Denoising Diffusion Implicit Models (DDIM) with state propagation mechanisms to recover missing trajectory segments from partially observed spatiotemporal data.

## Overview

Trace addresses the trajectory recovery problem by formulating it as a conditional generation task using diffusion models. The key innovation is the **State Propagation Mechanism** that maintains temporal dependencies across diffusion steps, enabling more accurate trajectory reconstruction.

### Key Features

- **Diffusion-Based Recovery**: Uses DDIM/DDPM for generating missing trajectory segments
- **State Propagation**: Maintains hidden states across diffusion timesteps for better temporal consistency
- **Multi-Architecture Support**: Multiple model variants for different trajectory encoding strategies
- **Flexible Masking**: Supports variable-rate trajectory masking (20%-90% erasure)
- **Multi-Dataset Support**: Works with both apartment delivery and taxi trajectory datasets

## Architecture

The model consists of three main components:

1. **Trace (U-Net Backbone)**: A U-Net-based denoising network that predicts noise at each diffusion step
2. **Linkage Module**: Propagates hidden states between consecutive diffusion timesteps
3. **Embedder** (for apartments dataset): Encodes metadata features for conditional generation

### Model Variants

- **Trace_MultiSeq_Add**: Multi-sequence encoding with additive state propagation (default)
- **Trace_MultiSeq_Cat**: Multi-sequence encoding with concatenative state propagation
- **Trace_MultiSeq_CA**: Multi-sequence encoding with cross-attention
- **Trace_MultiVec_Add**: Multi-vector encoding with additive propagation
- **Trace_Seq_Cat**: Single-sequence encoding with concatenation

## Project Structure

```
.
├── Configs.py                      # Configuration file for models, datasets, and training
├── train.py                        # Main training script
├── eval.py                         # Evaluation and visualization utilities
├── CreateTestSet.py                # Script to create test datasets
├── Utils.py                        # Helper functions (loss, model I/O, etc.)
├── EvalUtils.py                    # Evaluation metrics utilities
├── BatchManagers/                  # Data loading and batch scheduling
│   ├── BatchManagerApartments.py   # Batch manager for apartment dataset
│   ├── BatchManagerTaxi.py         # Batch manager for taxi dataset
│   └── ThreadedScheduler.py        # Multi-threaded data loading
├── Dataset/                        # Dataset files and loaders
│   ├── DatasetApartments.py        # Apartment delivery trajectory dataset
│   ├── DatasetTaxi.py              # Taxi trajectory dataset (Xian/Chengdu)
│   ├── apartment_dataset.pth       # Pre-processed apartment data
│   ├── LinkToTaxiDataset.md        # Instructions to download taxi data
│   └── test_*.pth                  # Test set files
├── DDM/                            # Diffusion model implementations
│   ├── DDIM.py                     # Denoising Diffusion Implicit Model
│   └── DDPM.py                     # Denoising Diffusion Probabilistic Model
└── Models/                         # Model architectures
    ├── Basics.py                   # Basic building blocks
    ├── EmbeddingModule.py          # Metadata embedding module
    ├── StateProp.py                # State propagation modules
    ├── Trace_MultiSeq_Add.py       # Multi-sequence additive model
    ├── Trace_MultiSeq_Cat.py       # Multi-sequence concatenative model
    ├── Trace_MultiSeq_CA.py        # Multi-sequence cross-attention model
    ├── Trace_MultiVec_Add.py       # Multi-vector additive model
    └── Trace_Seq_Cat.py            # Single-sequence model
```

## Requirements

The project requires the following dependencies:

```
torch>=2.0.0
matplotlib
numpy
tqdm
rich
tensorboard
einops
```

## Paper

This work has been published at the **ACM Web Conference 2026 (WWW '26)**.

**Conference Details:**
- **Event**: Proceedings of the ACM Web Conference 2026 (WWW '26)
- **Date**: April 13-17, 2026
- **Location**: Dubai, United Arab Emirates
- **DOI**: [10.1145/3774904.3792461](https://doi.org/10.1145/3774904.3792461)
- **ISBN**: 979-8-4007-2307-0/2026/04

For more details about the methodology and results, please refer to the paper.

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International).

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{trace2026,
  title={TRACE: Trajectory Recovery with State Propagation Diffusion for Urban Mobility},
  author={[Authors]},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={[Pages]},
  year={2026},
  month={April},
  address={Dubai, United Arab Emirates},
  doi={10.1145/3774904.3792461},
  isbn={979-8-4007-2307-0/2026/04},
  publisher={Association for Computing Machinery},
  series={WWW '26}
}
```
