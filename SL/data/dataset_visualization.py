import os
os.environ["HF_HOME"] = "/mnt/ssd/huggingface"

from renumics import spotlight

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

dataset_path = "HuggingFaceFW/fineweb"
dataset_name = dataset_path.split("/")[-1]

fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train[11478528:11578528]")

# apparently spotlight tries to follow urls automatically, 
# which does not work here
fw = fw.remove_columns("url")

spotlight.show(fw)