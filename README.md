## Attacking Neural Text Detectors

Code for "Attacking Neural Text Detectors" (https://arxiv.org/abs/2002.11768).

Run ``python download_dataset.py`` to download the GPT-2 top k-40 neural text test set created by OpenAI. For more documentation regarding this and similar datasets, visit https://github.com/openai/gpt-2-output-dataset.

OpenAI RoBERTa neural text detector can be downloaded by running ``wget https://storage.googleapis.com/gpt-2/detector-models/v1/detector-large.pt``.

Install requirements via ``pip install -r requirements.txt``.

Run ``python main.py`` to run a sample experiment.

