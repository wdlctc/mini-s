# Extending Gemma-2-9B Training Context with Mini-Sequence Technology

In the ever-evolving landscape of artificial intelligence, training large language models (LLMs) with extended context lengths has become a critical challenge. Mini-sequence technology, introduced by Luo et al. (2024) in their paper "MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training," is a game-changing approach that's pushing the boundaries of what's possible in LLM training. This README explores how to apply this innovative technique to fine-tune Google's Gemma 2 9B model with extended context length.

## Table of Contents

- [What is Mini-Sequence Technology?](#what-is-mini-sequence-technology)
- [The Power of Mini-Sequence: Key Benefits](#the-power-of-mini-sequence-key-benefits)
- [Practical Application: Fine-tuning Google Gemma 2 9B](#practical-application-fine-tuning-google-gemma-2-9b)
  - [Setup and Requirements](#setup-and-requirements)
  - [Fine-tuning Process](#fine-tuning-process)
- [Conclusion](#conclusion)
- [References](#references)

## What is Mini-Sequence Technology?

Mini-sequence is an advanced memory optimization technique designed to tackle one of the most significant hurdles in training state-of-the-art language models: managing the enormous memory requirements for processing long sequences of text. By partitioning input sequences into smaller, more manageable chunks, mini-sequence allows for efficient processing of much longer contexts than traditional methods.

## The Power of Mini-Sequence: Key Benefits

- **Extended Context Lengths**: Mini-sequence enables training on sequences up to 4-12 times longer than standard implementations, dramatically increasing the model's ability to understand and generate coherent long-form content.
- **Memory Efficiency**: By optimizing memory usage, mini-sequence allows researchers and developers to train larger models or use longer sequences on existing hardware.
- **Maintained Performance**: Despite its memory-saving capabilities, mini-sequence maintains comparable training throughput to standard methods, ensuring efficiency doesn't come at the cost of speed.
- **Scalability**: The technique works well with distributed training setups, allowing for linear scaling of sequence length with the number of GPUs used.

## Practical Application: Fine-tuning Google Gemma 2 9B

Let's walk through the process of fine-tuning the Google Gemma 2 9B model with an 8,192 token context length using mini-sequence technology. We'll use an NVIDIA H100 GPU for this process.

### Setup and Requirements

Follow these steps to set up your environment:

1. Obtain one H100 NVL GPU (available on vastAI)

2. Clone the LongLoRA repository and install dependencies:

   ```bash
   git clone https://github.com/dvlab-research/LongLoRA
   cd LongLoRA
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   ```

3. Deploy the mini-sequence version of Hugging Face Transformers:

   ```bash
   pip install -U git+https://github.com/wdlctc/transformers
   ```

### Fine-tuning Process

Before running the fine-tuning script, set an environment variable to clean memory fragments:

```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:516"
```

Now, run the fine-tuning script:

```bash
python fine-tune.py  \
    --model_name_or_path google/gemma-2-9b \
    --bf16 True \
    --output_dir path_to_saving_checkpoints \
    --cache_dir path_to_cache \
    --model_max_length 8192 \
    --use_flash_attn True \
    --low_rank_training False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --low_rank_training False \
    --max_steps 1000
```

This script demonstrates how mini-sequence allows us to fine-tune the Google Gemma 2 9B model with a context length of 8,192 tokens, which is a significant improvement over standard training methods.

## Conclusion

Mini-sequence technology is revolutionizing the way we train and fine-tune large language models like Google's Gemma 2 9B. By enabling the processing of longer context lengths while maintaining efficiency, it opens up new possibilities for creating more capable and context-aware AI systems. Whether you're a researcher pushing the boundaries of AI or a developer looking to enhance your language models, mini-sequence is a powerful tool that deserves a place in your toolkit.

As we continue to explore the frontiers of AI, techniques like mini-sequence will play a crucial role in unlocking the full potential of large language models. Stay tuned for more developments in this exciting field!

## References

Luo, C., Zhao, J., Chen, Z., Chen, B., & Anandkumar, A. (2024). MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training. *arXiv preprint arXiv:2407.15892*.

For more details on the mini-sequence technology, please refer to the original paper:

```bibtex
@misc{luo2024mst,
      title={MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training}, 
      author={Luo, Cheng and Zhao, Jiawei and Chen, Zhuoming and Chen, Beidi and Anandkumar, Anima},
      year={2024},
      eprint={2407.15892},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
