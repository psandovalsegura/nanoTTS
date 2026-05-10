import string
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

# 0. load the normalized transcripts from the splits we'll be using for training
split_names = ['train-clean-100', 'train-clean-360']
with open(f"{split_names[0]}.json", "r") as f:
    split0 = json.load(f)

with open(f"{split_names[1]}.json", "r") as f:
    split1 = json.load(f)

all_transcripts = split0["transcripts"] + split1["transcripts"] # list of 33236+116500 normalized transcripts in libritts split_names
print(f"Loaded {len(all_transcripts)} transcripts from {split_names}")

# 1. core special tokens. <UNK> is required by the BPE model. 
core_specials = ["<UNK>", "<PAD>", "<BOS>", "<EOS>", "<AUDIO_START>"]

# 2. initialize a BPE Tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

# 3. configure the Trainer for exactly 2048 tokens, starting with the alphabet, digits, punctuation, and metaspace marker
LIBRITTS_VOCAB_SIZE = 2048 
alphabet = list(string.ascii_letters + string.digits + string.punctuation + "▁")

trainer = BpeTrainer(
    vocab_size=LIBRITTS_VOCAB_SIZE,
    initial_alphabet=alphabet,
    special_tokens=core_specials,
    show_progress=True
)

tokenizer.train_from_iterator(all_transcripts, trainer=trainer)
tokenizer.save("libritts_bpe.json")

# ==========================================
# Tests
# ==========================================
assert tokenizer.get_vocab_size(with_added_tokens=False) == LIBRITTS_VOCAB_SIZE, f'text vocab size should be {LIBRITTS_VOCAB_SIZE}'
assert tokenizer.get_vocab_size(with_added_tokens=True) == LIBRITTS_VOCAB_SIZE, f'total vocab size should be {LIBRITTS_VOCAB_SIZE}'
assert tokenizer.token_to_id("▁") is not None, "metaspace marker should be in vocab"

test_string = f"<BOS>Hello world!<AUDIO_START><EOS><PAD><PAD>"
encoded = tokenizer.encode(test_string)
print(f"Encoded: {encoded.tokens}")
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")