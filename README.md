# microGPT.lua

A compact, dependency-free implementation of a GPT (Generative Pre-trained Transformer) model in pure Lua. This project is a direct port of Karpathy's minimal Python GPT, designed to illustrate the core algorithm in its most atomic form.

## Features

-   **Pure Lua:** No external libraries or dependencies required, making it highly portable.
-   **Autograd Engine:** Includes a custom `Value` class for automatic differentiation, enabling training.
-   **GPT Architecture:** Implements a simplified GPT model with multi-head attention and MLP blocks.
-   **Adam Optimizer:** Utilizes the Adam optimization algorithm for efficient model training.
-   **Training & Inference:** Supports a training loop with cosine learning rate decay and a sampling mechanism for text generation.
-   **Dataset Handling:** Automatically downloads `names.txt` for training.

## Usage

To run this microGPT, simply execute the `gpt.lua` file using a Lua interpreter:

```bash
lua gpt.lua
```

The script will automatically download the required `input.txt` dataset if not present, train the model, and then perform inference to generate samples.

## Origin

This project is a Lua port of the educational GPT implementation by Andrej Karpathy. The original Python version can be found [here](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
