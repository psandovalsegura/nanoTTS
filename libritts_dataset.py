import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TTSDataset(Dataset):
    def __init__(self, libritts_dataset, tokenizer):
        self.dataset = libritts_dataset
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id
        self.astart_id = tokenizer.audio_start_id
        self.in_eos_id = tokenizer.in_eos_id
        self.out_eos_id = tokenizer.out_eos_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # LibriTTS: waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id
        waveform, sr, _, normalized_text, speaker_id, _, _ = self.dataset[idx]

        prompt_string = f"<BOS>{normalized_text}<SPK_{speaker_id}><AUDIO_START>"
        text_ids = self.tokenizer.encode_text(prompt_string)
        audio_ids = self.tokenizer.encode_audio(waveform)

        sequence = text_ids + audio_ids + [self.in_eos_id]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)

        # shift X and Y for NTP
        X = sequence_tensor[:-1]
        Y = sequence_tensor[1:].clone()

        # mask out text positions in target so loss only depends on predicting audio tokens
        astart_idx = (X == self.astart_id).nonzero(as_tuple=True)[0].item()
        Y[:astart_idx] = -100 

        # our model only predicts audio tokens, so target audio token ids should be in the range [0, audio_vocab_size)
        Y[Y >= self.tokenizer.audio_offset] -= self.tokenizer.audio_offset
        # remap input space <EOS> to output space <EOS>, which is the last audio token id
        Y[Y == self.in_eos_id] = self.out_eos_id
        return X, Y

def create_collate_fn(pad_id):
    def collate_fn(batch):
        xs, ys = zip(*batch)
        # pad X with the BPE's <PAD> id
        # pad Y with -100 so padding doesn't affect the loss
        x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        y_padded = pad_sequence(ys, batch_first=True, padding_value=-100)
        return x_padded, y_padded
    return collate_fn
