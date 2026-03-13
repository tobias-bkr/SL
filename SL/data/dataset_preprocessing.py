import os
os.environ["HF_HOME"] = "/mnt/ssd/huggingface"

from itertools import chain

import numpy as np

from datasets import load_dataset
from datasets import Dataset
from tokenizers import Tokenizer

# ========== Config ==========

# name of the dataset on huggingface; 
# downloaded already IN("roneneldan/TinyStories", "HuggingFaceFW/fineweb")
# or local path to a huggingface dataset (in that case dataset_version probably needs to be "default")
# or local path to a .txt
dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_is_file = False # set True only if the dataset_name is a local .txt file path
dataset_save_path = "./SL/data/fineweb-edu_preprocessed"
dataset_version="sample-10BT"
dataset_split = "train"
text_column_name = "text"
cast_to_string = False
tokenizer = Tokenizer.from_file("SL/tokenizers/fineweb-edu_tokenizer_1.json")
seq_len = 256 # seq_len that will be used for training
delimiter = 5 # id to give the delimiter, should be [DOC] token in the tokenizer

# ========== Functions ==========

def tokenize_group_and_separate(examples):
    global seq_len
    global delimiter
    global cast_to_string
    global text_column_name

    if(cast_to_string == True):
        examples[text_column_name] = [str(example) for example in examples[text_column_name]]

    chunks = tokenizer.encode_batch(examples[text_column_name])
    ids = list(chain.from_iterable((chunk.ids + [delimiter]) for chunk in chunks))
    arr = np.array(ids, dtype=np.int32)
    # number of full sequences
    n_full = len(arr) // (seq_len + 1)
    # discards remainder of every batch (should be << 2% total)
    trimmed = arr[: n_full * (seq_len + 1)]
    processed = trimmed.reshape(n_full, seq_len + 1)

    return {"text": processed.tolist()}

# ========== Preprocessing ==========

if(dataset_is_file):
    # creates a one entry dataset of the whole text file
    with open(dataset_name) as dsTextFile:
        dsText = dsTextFile.read()
    dataset = Dataset.from_dict({"text": [dsText]})
else:
    dataset = load_dataset(dataset_name, name=dataset_version,
                  split=dataset_split, keep_in_memory=False)

dataset = dataset.map(
    tokenize_group_and_separate,
    batched=True,
    batch_size=4096,              # larger batches = less overhead
    remove_columns=dataset.column_names,
    num_proc=8,                   # match physical cores
    load_from_cache_file=True,
    writer_batch_size=32768,    # reduce I/O syncs
)
 
dataset.save_to_disk(dataset_save_path)
