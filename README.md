# Visual Language Model for Remote Sensing 🌍📡

This project implements a **Visual Question Answering (VQA)** system for satellite and aerial imagery. Inspired by [LLaVA: Visual Instruction Tuning](https://llava-vl.github.io), it incorporates a customized architecture using **ChatGPT2** as the language model and is fine-tuned on the **EarthVQA dataset**.

---

## 🧠 Architecture Summary

Your model is composed of:

- **Vision Encoder**: `CLIPViT-B32` from OpenAI
- **Projection Layer**: A linear+LayerNorm module that maps image features into the GPT2 token embedding space
- **Language Model**: `GPT2LMHeadModel` (from Hugging Face Transformers)

A custom module `VisionTextModel` combines these three, where image features are embedded and concatenated to the input token stream for GPT2.

---

## 🧪 Training Workflow

### 📍 Pretraining Stage (Not in this notebook)
- Trained the projection layer to map CLIP image embeddings to token embedding space.
- Dataset: **Stanford Image Captioning Dataset**
- Loss: Cross-entropy over predicted captions

### 📍 Fine-tuning Stage (this notebook)
- Dataset: **EarthVQA** CSV with image paths, questions, and answers
- Batch size: `8`
- Epochs: `5`
- Optimizer: `AdamW` (`lr=5e-5`)
- Loss Function: `GPT2LMHeadModel`'s built-in language modeling loss
- Gradient accumulation: `2 steps`
- Clip Gradients: Yes (max norm = `1.0`)
- Multi-GPU: `torch.nn.DataParallel` used if available

```python
loss = outputs.loss.mean()  # averaged over GPUs
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
