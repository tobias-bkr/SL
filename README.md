This is a pytorch library for supervised machine learning, specifically LLM pretraining.
modules.py contains custom pytorch code for the transformer architecture as well as a simple integration for flash attention.
train.py allows pretraining the transformer architecture on tokenized datasets, which can be created from any hugging face dataset using
tokenizer_from_dataset.py and dataset_preprocessing.py for creating a tokenizer and tokenizing the entire dataset respectively.

The project uses uv as its project manager.
To install the required dependencies run 'uv sync' in the directory of the cloned repository.