"""Runs every cell of the Colab notebook here, except:
  [COLAB-ONLY] cells (upload/download), and
  [MODEL-LOAD] cell, replaced by a tiny stand-in GPT-2 + local tokenizer
               (so the test runs offline, without downloading GPT-2).
"""
import json, os, shutil, sys, tempfile
WM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the wm/ folder
nb_path = os.path.join(WM, "Watermark_GPT2_Colab.ipynb")
work = os.path.join(tempfile.gettempdir(), "nbtest")
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
shutil.copy(os.path.join(WM, "data", "human_texts.csv"), work)
os.chdir(work); sys.path.insert(0, work)

STANDIN = '''
import torch, pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
d = pd.read_csv("human_texts.csv")
tk = Tokenizer(models.WordLevel(unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tk.decoder = decoders.WordPiece(prefix="##")
tk.train_from_iterator(list(d.prompt)+list(d.human_text), trainers.WordLevelTrainer(vocab_size=8000, special_tokens=["[UNK]","[EOS]"]))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="[EOS]")
torch.manual_seed(0)
model = GPT2LMHeadModel(GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=512, n_embd=64, n_layer=2, n_head=2,
        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.eos_token_id)).eval()
V = model.config.vocab_size
'''
nb = json.load(open(nb_path))
g = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if s.startswith("# [COLAB-ONLY]"):
        continue
    if s.startswith("# [MODEL-LOAD]"):
        s = STANDIN
    if s.startswith("%%writefile"):
        name, body = s.split("\n", 1)
        open(name.split()[1], "w").write(body + "\n")
        continue
    s = s.replace("N_PROMPTS = 498", "N_PROMPTS = 40")
    exec(compile(s, f"cell{i}", "exec"), g)
print("NOTEBOOK RAN OK; outputs:", sorted(os.listdir("outputs")))
