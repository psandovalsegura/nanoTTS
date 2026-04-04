from tokenizers import Tokenizer


class JointTokenizer:
    def __init__(self, text_tokenizer, wav_tokenizer):
        self.text_tokenizer = text_tokenizer
        self.wav_tokenizer = wav_tokenizer

        self.pad_id = self.text_tokenizer.token_to_id("<PAD>")
        self.audio_start_id = self.text_tokenizer.token_to_id("<AUDIO_START>")
        self.in_eos_id = self.text_tokenizer.token_to_id("<EOS>")

        self.text_vocab_size = self.text_tokenizer.get_vocab_size(with_added_tokens=True)
        self.audio_vocab_size = self.wav_tokenizer.feature_extractor.encodec.quantizer.bins
        self.audio_offset = self.text_vocab_size
        self.in_vocab_size = self.text_vocab_size + self.audio_vocab_size
        self.out_vocab_size = self.audio_vocab_size + 1 # @psando: +1 for EOS
        self.out_eos_id = self.audio_vocab_size         # @psando: last output id

        assert self.pad_id is not None
        assert self.audio_start_id is not None
        assert self.in_eos_id is not None
        assert self.out_eos_id is not None
        assert self.audio_vocab_size > 0

    def encode_text(self, text):
        return self.text_tokenizer.encode(text).ids

    def encode_audio(self, waveform):
        _, audio_ids = self.wav_tokenizer.encode_infer(
            waveform,
            bandwidth_id=self.wav_tokenizer.bandwidth_id,
        )
        raw_audio_ids = audio_ids.squeeze().tolist()
        return [audio_id + self.audio_offset for audio_id in raw_audio_ids]

    def decode(self, sequence):
        # sequence is expected to be in input space
        assert sequence.shape[0] == 1, "batch size must be 1 for inference"
        assert sequence.ndim == 2, "input sequence should have shape (1, seq_len)"

        # get generated audio ids by finding audio start token and taking everything after it
        audio_start_idx = (sequence == self.audio_start_id).nonzero(as_tuple=True)[1].item()
        audio_ids = sequence[:, audio_start_idx + 1 :]

        # find earliest input <EOS> token. if it doesn't exist, use the full generated sequence
        eos_mask = audio_ids == self.in_eos_id
        if eos_mask.any():
            eos_idx = eos_mask.nonzero(as_tuple=True)[1].min().item()
            audio_ids = audio_ids[:, :eos_idx]

        # generated sequence is in input space, so convert audio tokens back to output space
        audio_ids = audio_ids - self.audio_offset
        features = self.wav_tokenizer.codes_to_features(audio_ids)
        return self.wav_tokenizer.decode(
            features,
            bandwidth_id=self.wav_tokenizer.bandwidth_id,
        )


def create_joint_tokenizer(tokenizer_path, wav_tokenizer):
    text_tokenizer = Tokenizer.from_file(tokenizer_path)
    return JointTokenizer(text_tokenizer=text_tokenizer, wav_tokenizer=wav_tokenizer)
