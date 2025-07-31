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

### 📍 Pretraining Stage 
- Trained a projection layer to map CLIP image embeddings into the GPT-2 token embedding space.
- Frozen components: CLIP Vision Encoder (`openai/clip-vit-base-patch32`) and GPT-2 Language Model.
- Dataset: **Stanford Image Paragraph Captioning Dataset**
- Input format: Prompted captions in the form  
  `User: <instruction>\nAssistant: <caption>`
- Projection: Linear + LayerNorm mapping to 32 GPT-equivalent tokens.
- Loss: Cross-entropy over caption tokens (image tokens ignored).
- Training Setup:
  - Batch Size: 8
  - Epochs: 3
  - Gradient Accumulation: 4 steps
  - Optimizer: AdamW with learning rate `1e-4`
  - Precision: Float32

### 📍 Fine-tuning Stage
- Dataset: **EarthVQA** CSV with image paths, questions, and answers
- Batch size: `8`
- Epochs: `5`
- Optimizer: `AdamW` (`lr=5e-5`)
- Loss Function: `GPT2LMHeadModel`'s built-in language modeling loss
- Gradient accumulation: `2 steps`
- Clip Gradients: Yes (max norm = `1.0`)
- Multi-GPU: `torch.nn.DataParallel` used if available

