# Fine-Tuning LLM with Unsloth

This project demonstrates how to **fine-tune a Large Language Model (LLM)** using [Unsloth](https://github.com/unslothai/unsloth) — a fast and memory-efficient tool built on top of Hugging Face's ecosystem, optimized for low-resource environments.

## Why Fine-Tune?

* **Add New Knowledge**: Extend model capabilities with domain-specific or updated data.
* **Improve Performance**: Boost accuracy on custom tasks (e.g., classification, summarization).
* **Enable RAG**: Fine-tuned models integrate better with Retrieval-Augmented Generation pipelines.
* **Task Specialization**: Tailor LLMs for unique use-cases like healthcare, education, customer support, etc.

## Common Approaches to Adapt LLMs

1. **Pretraining**

   * Expensive and compute-heavy
   * Requires massive datasets
   * Used to create base foundation models

2. **Full Fine-Tuning**

   * Updates *all* model parameters
   * High compute/memory usage
   * Better for very large custom datasets

3. **LoRA / QLoRA (Parameter-Efficient Fine-Tuning)**

   * Only fine-tunes a small portion of the model
   * Memory efficient (ideal for laptops/Colab)
   * Perfect for low-cost, fast experimentation

---

##  What This Repo Does

* Loads a Mistral-7B model using `Unsloth`
* Applies **LoRA** using `FastLanguageModel.get_peft_model()`
* Trains using `SFTTrainer` on a custom dataset
* Optimized with:

  * 8-bit model loading
  * Gradient checkpointing
  * Mixed precision (fp16 / bf16)

## Tools & Stack

* `unsloth`
* `transformers`
* `peft`
* `trl` (for SFTTrainer)
* `datasets` (HuggingFace)


## Output

* Fine-tuned model saved to `unsloth-test/`
* Ready for inference or further training

