# MINI-SEQUENCE TRANSFORMER (MST)

## Overview

MINI-SEQUENCE TRANSFORMER (MST) is a simple and effective method for highly efficient and accurate LLM training with extremely long sequences. Our research demonstrates that the Llama3-8B model can be trained with context lengths up to 60k tokens on a single NVIDIA A100 GPU with 80GB memory, representing a 12x increase in maximum sequence length compared to standard implementations.

We believe that our work opens new avenues for long-sequence training of LLMs, and reduces the hardware obstacles for researchers and developers aiming to create LLMs with long context.

![mst](./doc/img/mst.png)

## Key Features

- Enables training Llama3-8B with 60k token sequences on a single A100 GPU (4x longer than activation recomputation alone)
- Maintains the same training throughput as standard implementations
- Fully general and implementation-agnostic, supporting most parameter-efficient training methods
- Easy to integrate into existing training frameworks with minimal code changes

## Installation

To install and run the mini-s model, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/your-username/mini-s.git
   cd mini-s
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Install the `flash-attn` package:

   ```
   pip install flash-attn --no-build-isolation
   ```

Note: The `--no-build-isolation` flag is used to avoid potential build conflicts.

## Usage

To run the benchmark script and evaluate the performance of the mini-s model, use the following one-click-run command:
   ```
   python benchmark_replace.py
   ```
This file contains the modifications made to the original model to create the mini-s version.

## How It Works

MST partitions input sequences and iteratively processes mini-sequences to reduce intermediate memory usage. When integrated with activation recomputation, this allows for significant memory savings in both forward and backward passes.

## Benefits

MST opens up new possibilities for training LLMs on long sequences using limited hardware resources:

- Enables efficient training on much longer sequences
- Improves LLM capabilities across tasks that benefit from extended context, like long document summarization and multi-turn dialogue
- Requires no changes to model architecture, making it broadly applicable to a wide range of existing and future transformer models

## Experimental Results

We evaluated MST on popular models like Llama3-8B and Llama2-7B. In our experiments, we observed:

- No degradation in convergence or throughput even with 12x longer sequences compared to standard implementations
- Llama3-8B can be trained with context lengths up to 60k tokens on a single NVIDIA A100 GPU
- Llama2-7B can be trained with context lengths up to 84k tokens on a single NVIDIA A100 GPU

## Contributing

We welcome contributions to the MST project. If you're interested in collaborating on MST research or have questions about our work, please open an issue or submit a pull request.

## License

This project is licensed under, MIT License.

## Contact

For any inquiries, please contact wdlctc@gmail.com.