import torch
from collections import OrderedDict

weight_path = "SL/checkpoints/fineweb_transfomer_max_v8/d2025-10-02|18:44:24_s0/model.pt"
compile_prefix = "_orig_"
old_state_dictionary = torch.load(weight_path)
new_state_dictionary = OrderedDict()

for key in old_state_dictionary:
    # add the prefix to every key
    new_state_dictionary[compile_prefix+key] = old_state_dictionary[key]

torch.save(new_state_dictionary, f=weight_path)
