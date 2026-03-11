import string
import pickle
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 0. load the transcripts and speaker IDs from the previous step
with open("libritts_tokenizer_data.pkl", "rb") as f:
    data = pickle.load(f)

all_transcripts = data["all_transcripts"] # list of 33236 normalized transcripts in libritts 'train-clean-100' subset
speaker_ids = data["speaker_ids"]         # set of 247 unique speaker IDs in libritts 'train-clean-100' subset

# 1. core special tokens. <UNK> is required by the BPE model. 
core_specials = ["<UNK>", "<PAD>", "<BOS>", "<EOS>", "<AUDIO_START>"]

# 2. initialize a BPE Tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

# 3. configure the Trainer for exactly 2048 tokens after adding the speaker tokens
LIBRITTS_VOCAB_SIZE = 2048 
alphabet = list(string.ascii_letters + string.digits + string.punctuation)

trainer = BpeTrainer(
    vocab_size=LIBRITTS_VOCAB_SIZE - len(speaker_ids),
    initial_alphabet=alphabet,
    special_tokens=core_specials,
    show_progress=True
)

tokenizer.train_from_iterator(all_transcripts, trainer=trainer)

# 5. create and append the speaker tokens. sort to ensure the mapping is consistent every time
sorted_spk_ids = sorted(list(speaker_ids))
speaker_tokens = [f"<SPK_{spk}>" for spk in sorted_spk_ids]
tokenizer.add_special_tokens(speaker_tokens)
tokenizer.save("libritts_bpe.json")

# ==========================================
# Tests
# ==========================================
assert tokenizer.get_vocab_size(with_added_tokens=False) == LIBRITTS_VOCAB_SIZE - len(speaker_ids), f'text vocab size should be {LIBRITTS_VOCAB_SIZE - len(speaker_ids)}'
assert tokenizer.get_vocab_size(with_added_tokens=True) == LIBRITTS_VOCAB_SIZE, f'total vocab size should be {LIBRITTS_VOCAB_SIZE}'
assert tokenizer.token_to_id(speaker_tokens[0]) == LIBRITTS_VOCAB_SIZE - len(speaker_ids), f'first speaker token ID should be {LIBRITTS_VOCAB_SIZE - len(speaker_ids)}'

test_string = f"<BOS>Hello world!<SPK_19><AUDIO_START><EOS><PAD><PAD>"
encoded = tokenizer.encode(test_string)
print(f"Encoded: {encoded.tokens}")