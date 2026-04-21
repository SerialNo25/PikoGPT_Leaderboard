#!/usr/bin/env python3
"""
PikoGPT - A tiny language model built from scratch.

Usage:
    python main.py --stage data-preprocessing
    python main.py --stage data-preprocessing --sft
    python main.py --stage pretraining --checkpoint checkpoints/latest.pt
    python main.py --stage sft --checkpoint runs/run_xxx/best.pt --mode all
    python main.py --stage evaluation --checkpoint runs/run_xxx/best.pt --benchmarks lambada hellaswag arc
    python main.py --stage inference --checkpoint checkpoints/final.pt --prompt "Hello"
    python main.py --list-stages

Distributed Training:
    python main.py --stage pretraining --num-gpus 4
    python main.py --stage sft --checkpoint best.pt --num-gpus 4 --mode all
    python main.py --stage pretraining --gpu-ids 0,1,2,3
"""

