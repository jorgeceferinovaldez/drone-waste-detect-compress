# drone-waste-detect-compress

Deep learning pipeline for coastal waste detection: edge compression with CompressAI, transmission, reconstruction and YOLO detection. MSc thesis · UNPA · Golfo San Jorge, Patagonia.

---

## Overview

This project proposes an architecture for coastal waste monitoring using drones. Instead of transmitting raw images, frames are compressed at the edge device using a deep learning model (`cheng2020-anchor` from CompressAI), the compressed bytes are transmitted over the network, and the server reconstructs the image before running YOLO detection.

The central experiment compares **throughput** between direct image transmission (baseline) and latent representation transmission (proposed system), measuring bytes transferred, latency, reconstruction quality (PSNR, SSIM) and detection accuracy (mAP).

```
Drone → Edge device (CompressAI encoder) → Network → Server (CompressAI decoder → YOLO) → Detections
```

## Waste classes

`plastico_rigido` · `film_bolsa` · `poliestireno` · `metal` · `vidrio` · `red_pesca` · `otros`

## Hardware

| Role | Device |
|---|---|
| Image capture | DJI Mini 4 Pro |
| Video capture | Elgato Cam Link 4K |
| Edge device | MacBook Pro M4 Pro · 24 GB |
| Server / training | Ryzen 9 5900X · RTX 4070 Ti Super 16 GB · 32 GB RAM · Pop OS 22.04 |

## Stack

- Python 3.10.12
- PyTorch 2.3.1
- CompressAI 1.2.6 — `cheng2020-anchor` (q=1 to q=6)
- Ultralytics YOLOv8 / v11
- Optuna 3.6.1
- TensorBoard 2.17.1

## Project structure

```
drone-waste-detect-compress/
├── data/               # raw, interim and processed datasets
├── models/             # trained CompressAI and YOLO checkpoints
├── notebooks/          # step-by-step experiment notebooks
├── reports/            # figures, metrics and TensorBoard logs
├── src/                # source code (CompressAI model, YOLO model, utils)
├── references/         # full project documentation
└── CLAUDE.md           # context file for Claude Code
```

Full documentation: [`references/README_COMPLETO.md`](references/README_COMPLETO.md)

## License

- Code: [MIT License](LICENSE.txt)
- Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

*Maestría en Informática y Sistemas · UNPA · Caleta Olivia, Santa Cruz, Argentina*  
Esp. Ing. Jorge Ceferino Valdez · 2024-2025
