import os
os.environ["HF_HOME"] = "/mnt/ssd/huggingface"

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# ========== Config ==========

# name of the dataset on huggingface; 
# downloaded already IN("roneneldan/TinyStories", "HuggingFaceFW/fineweb")
# or local path to a huggingface dataset (in that case dataset_version probably needs to be "default")
# or local path to a .txt
dataset_name = "mnt/ssd/huggingface/datasets/roneneldan___tiny_stories" 
dataset_is_file = False # set True only if the dataset_name is a local .txt file path
dataset_version="default"
dataset_split = "train"

# vocab_size is the minimum, every character in the dataset gets at least one token
vocab_size = 2048
# only allow english chars
allowed_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()\n")
# ordering determines token id, starting at 0
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[DOC]"]

tokenizer_save_path = "./SL/tokenizers/TinyStories_tokenizer_3.json"
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# pre tokenizer that sets boundaries at whitespaces, for subword tokenization
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
# standard BytePairEncoding Trainer
trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    initial_alphabet=allowed_chars
)

# =========== Train Tokenizer ===========

if(dataset_is_file):
    # will load the whole dataset into memory!
    tokenizer.train(dataset_name, trainer)
else:
    # streaming=False would be faster but loads whole dataset into memory
    dataset = load_dataset(dataset_name, name=dataset_version,
                    split=dataset_split, streaming=True)
    tokenizer.train_from_iterator(dataset, trainer=trainer)
tokenizer.save(tokenizer_save_path)
