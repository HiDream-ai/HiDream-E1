# HiDream-E1

![HiDream-E1 Demo](assets/demo.jpg)

HiDream-E1 is an image editing model built on [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1).

<!-- ![Overview](demo.jpg) -->
<span style="color: #FF5733; font-weight: bold">For more features and to experience the full capabilities of our product, please visit [https://vivago.ai/](https://vivago.ai/).</span>

## Project Updates
- 🌟 **July 16, 2025**: We've open-sourced the updated image editing model **HiDream-E1.1**. It supports dynamic resolution and is better in image quality and editing accuracy compared to HiDream-E1-Full.
- 📝 **May 28, 2025**: We've released our technical report [HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer](https://arxiv.org/abs/2505.22705).  Please use the Bibtex below to cite the paper.
- 🚀 **April 28, 2025**: We've open-sourced the instruction-based image editing model **HiDream-E1**. 


## Models

We offer the full version of HiDream-E1. For more information about the models, please refer to the link under Usage.

| Name            | Script                                             | Inference Steps | Resolution | HuggingFace repo       |
| --------------- | -------------------------------------------------- | --------------- | ---------- | ---------------------- |
| HiDream-E1-Full | [inference.py](./inference.py)                     | 28              | 768x768    | 🤗 [HiDream-E1-Full](https://huggingface.co/HiDream-ai/HiDream-E1-Full)  |
| HiDream-E1.1 | [inference_e1_1.py](./inference_e1_1.py)                     | 28              | Dynamic(1M pixels)    | 🤗 [HiDream-E1.1](https://huggingface.co/HiDream-ai/HiDream-E1-1)  |
> [!NOTE]
> The code and model are under development and will be updated frequently.


## Quick Start
Please make sure you have installed [Flash Attention](https://github.com/Dao-AILab/flash-attention) and latest [Diffusers](https://github.com/huggingface/diffusers.git). We recommend CUDA versions 12.4 for the manual installation.

```sh
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
pip install -U git+https://github.com/huggingface/diffusers.git
```


For HiDream-E1.1, you can run the following script to generate images:

``` python 
python ./inference_e1_1.py
```

For HiDream-E1-Full, you can run the following script to generate images:

``` python 
python ./inference.py
```

> [!NOTE]
> We add a refine_strength parameter to the pipeline to control the balance between editing and refinement stages. During the first (1 - refine_strength) portion of denoising steps, the model performs the main editing operation. The remaining refine_strength portion of steps uses HiDream-I1-Full for img2img refinement to enhance the final result. Set refine_strength to 0.0 to disable refinement.

> [!NOTE]
> The inference script will try to automatically download `meta-llama/Llama-3.1-8B-Instruct` model files. You need to [agree to the license of the Llama model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on your HuggingFace account and login using `huggingface-cli login` in order to use the automatic downloader.


> [!NOTE]
> For HiDream-E1.1, the model accepts direct instructions like `convert the image into a Ghibli style` and prompt refinement is no longer needed.
> 
> For HiDream-E1-Full, the model accepts instructions in the following format:
> ```
> Editing Instruction: {instruction}. Target Image Description: {description}
> ```
> 
> Example:
> ```
> Editing Instruction: Convert the image into a Ghibli style. Target Image Description: A person in a light pink t-shirt with short dark hair, depicted in a Ghibli style against a plain background.
> ```
> 
> To refine your instructions, use the provided script:
> ```bash
> python ./instruction_refinement.py --src_image ./test.jpeg --src_instruction "convert the image into a Ghibli style"
> ```
> 
> The instruction refinement script requires a VLM API key - you can either run vllm locally or use OpenAI's API.
> 


## Gradio Demo

We also provide a Gradio demo for interactive image editing. For HiDream-E1.1, you can run the demo with:

``` python
python gradio_demo_1_1.py 
```

For HiDream-E1-Full, you can run the demo with:

``` python
python gradio_demo.py 
```


## Evaluation Metrics

**Evaluation results on EmuEdit and ReasonEdit Benchmarks. Higher is better.**

| Model              | EmuEdit Global | EmuEdit Add  | EmuEdit Text | EmuEdit BG   | EmuEdit Color | EmuEdit Style | EmuEdit Remove | EmuEdit Local | EmuEdit Average | ReasonEdit |
|--------------------|----------------|--------------|--------------|--------------|---------------|---------------|----------------|---------------|-----------------|------------|
| OmniGen            | 1.37           | 2.09         | 2.31         | 0.66         | 4.26          | 2.36          | 4.73           | 2.10          | 2.67            | 7.36       |
| MagicBrush         | 4.06           | 3.54         | 0.55         | 3.26         | 3.83          | 2.07          | 2.70           | 3.28          | 2.81            | 1.75       |
| UltraEdit          | 5.31           | 5.19         | 1.50         | 4.33         | 4.50          | 5.71          | 2.63           | 4.58          | 4.07            | 2.89       |
| Gemini-2.0-Flash   | 4.87           | 7.71 | 6.30         | 5.10 | 7.30          | 3.33          | 5.94           | 6.29          | 5.99            | 6.95       |
| HiDream-E1         | 5.32 | 6.98         | 6.45 | 5.01         | 7.57 | 6.49 | 5.99 | 6.35 | 6.40 | 7.54 |
| HiDream-E1.1         | **7.47** | **7.97**         | **7.49** | **7.32** | **7.97** | **7.84** | **7.51** | **6.80** | **7.57** | **7.70** |

## License

The code in this repository and the HiDream-E1 models are licensed under [MIT License](./LICENSE).

## Citation

```bibtex
@article{hidreami1technicalreport,
  title={HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer},
  author={Cai, Qi and Chen, Jingwen and Chen, Yang and Li, Yehao and Long, Fuchen and Pan, Yingwei and Qiu, Zhaofan and Zhang, Yiheng and Gao, Fengbin and Xu, Peihan and others},
  journal={arXiv preprint arXiv:2505.22705},
  year={2025}
}
```
