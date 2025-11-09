# Learning Spectral Structure In-Context: Fourier Function Class Induction via Transformers

This repository extends the work from [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://github.com/dtsip/in-context-learning) to explore whether Transformers can induce spectral structures through in-context learning.

View the full paper [here](https://github.com/svemyh/in-context-learning-vertexai/blob/main/docs/Learning_Spectral_Structure_In-Context.pdf) or download it as a pdf directly [here](https://github.com/svemyh/in-context-learning-vertexai/raw/main/docs/Learning_Spectral_Structure_In-Context.pdf).

![](docs/setting.jpg)

## Abstract

In-context learning (ICL) allows Transformer models to perform new tasks by conditioning on input-output examples without updating parameters. While prior work has demonstrated that Transformers can emulate learning algorithms for simple function classes, it remains unclear whether they can induce structured transformations between complex domains, such as time and frequency. In this work, we explore the capacity of Transformers to in-context learn the Discrete Fourier Transform (DFT) by presenting paired time- and frequency-domain signal segments as prompts. We propose a synthetic task setup where each prompt consists of sampled signals and their spectral representations, and the model must predict the frequency structure of a new time-domain query. Our architecture builds on recent advances in patch-based tokenization and decoder-only Transformers to support spectrum regression and frequency classification tasks. We evaluate performance using metrics including frequency recovery accuracy, generalization to unseen frequencies, and prompt efficiency. Our results demonstrate that Transformers can recover spectral structure in context, offering a new perspective on learning structured signal transformations without gradient updates.

## Setup

This repository includes infrastructure-as-code using Terraform, Docker containerization for reproducible training environments, and Weights & Biases (WandB) integration for experiment tracking. Training jobs are deployed and executed on Google Cloud Vertex AI with GPU acceleration.

For detailed setup instructions, please see [SETUP.md](SETUP.md).

## Original Work

This work builds upon:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066

```bibtex
@InProceedings{garg2022what,
    title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
    author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
    year={2022},
    booktitle={arXiv preprint}
}
```

## Contributors

https://github.com/svemyh

https://github.com/ljs-233233

https://github.com/TonyHGF

https://github.com/JHJORE

https://github.com/Klovning


### Maintainers of the [Original Repository](https://github.com/dtsip/in-context-learning)
* [Shivam Garg](https://cs.stanford.edu/~shivamg/)
* [Dimitris Tsipras](https://dtsipras.com/)
