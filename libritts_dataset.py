import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

LIBRITTS_VOCAB_SIZE = 2048 
AUDIO_OFFSET = LIBRITTS_VOCAB_SIZE 
AUDIO_VOCAB_SIZE = 4096 # WavTokenizer has 4096 discrete codes
TOTAL_VOCAB_SIZE = LIBRITTS_VOCAB_SIZE + AUDIO_VOCAB_SIZE

class TTSDataset(Dataset):
    def __init__(self, libritts_dataset, bpe_tokenizer_path, wav_tokenizer):
        self.dataset = libritts_dataset
        self.text_tokenizer = Tokenizer.from_file(bpe_tokenizer_path)
        self.wav_tokenizer = wav_tokenizer
        
        self.pad_id = self.text_tokenizer.token_to_id("<PAD>")
        self.astart_id = self.text_tokenizer.token_to_id("<AUDIO_START>")
        self.eos_id = self.text_tokenizer.token_to_id("<EOS>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # LibriTTS: waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id
        waveform, sr, _, normalized_text, speaker_id, _, _ = self.dataset[idx]

        prompt_string = f"<BOS>{normalized_text}<SPK_{speaker_id}><AUDIO_START>"
        text_ids = self.text_tokenizer.encode(prompt_string).ids 

        _, audio_ids = self.wav_tokenizer.encode_infer(waveform, bandwidth_id=self.wav_tokenizer.bandwidth_id)
        raw_audio_ids = audio_ids.squeeze().tolist()
        audio_ids = [val + AUDIO_OFFSET for val in raw_audio_ids]

        sequence = text_ids + audio_ids + [self.eos_id]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)

        # shift X and Y for NTP
        X = sequence_tensor[:-1]
        Y = sequence_tensor[1:].clone()

        # mask out text positions in target so loss only depends on predicting audio tokens
        astart_idx = (X == self.astart_id).nonzero(as_tuple=True)[0].item()
        Y[:astart_idx] = -100 

        # our model only predicts audio tokens, so target audio token ids should be in the range [0, 4096)
        Y[Y >= AUDIO_OFFSET] -= AUDIO_OFFSET
        # we must also remap <EOS> token to be 4096
        Y[Y == self.eos_id] = AUDIO_VOCAB_SIZE

        return X, Y
    
    def _get_waveform_from_generated_sequence(self, sequence):
        assert isinstance(sequence, torch.Tensor)
        assert sequence.shape[0] == 1, "batch size must be 1 for inference"
        assert sequence.ndim == 2, "input sequence should have shape (1, seq_len)"
        # get generated audio tokens
        astart_idx = (sequence == self.astart_id).nonzero(as_tuple=True)[1].item()
        audio_ids = sequence[:, astart_idx+1:]

        # find earliest <EOS> token. if it doesn't exist, use the full generated sequence
        eos_mask = (audio_ids == self.eos_id) 
        if eos_mask.any():
            eos_idx = eos_mask.nonzero(as_tuple=True)[1].min().item()
            audio_ids = audio_ids[:, :eos_idx]

        # generated sequence is in input space, so convert audio tokens back to [0, 4095] range
        audio_ids -= AUDIO_OFFSET
        features = self.wav_tokenizer.codes_to_features(audio_ids)
        audio_out = self.wav_tokenizer.decode(features, bandwidth_id=self.wav_tokenizer.bandwidth_id)
        return audio_out

def create_collate_fn(pad_id):
    def collate_fn(batch):
        xs, ys = zip(*batch)
        # pad X with the BPE's <PAD> id
        # pad Y with -100 so padding doesn't affect the loss
        x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        y_padded = pad_sequence(ys, batch_first=True, padding_value=-100)
        return x_padded, y_padded
    return collate_fn