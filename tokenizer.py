import math
import tiktoken


class JointTokenizer:
    def __init__(self, wav_tokenizer):
        self.wav_tokenizer = wav_tokenizer

        self.EOS_TOKEN         = "<|endoftext|>"
        self.BOS_TOKEN         = "<|beginoftext|>"
        self.AUDIO_START_TOKEN = "<|audio|>"

        gpt2 = tiktoken.get_encoding("gpt2")
        # gpt2's only special, <|endoftext|>, is already at gpt2.eot_token;
        # place the new specials right after gpt2's existing range.
        special_tokens = {
            self.EOS_TOKEN:         gpt2.eot_token,
            self.BOS_TOKEN:         gpt2.n_vocab,
            self.AUDIO_START_TOKEN: gpt2.n_vocab + 1,
        }
        self.text_tokenizer = tiktoken.Encoding(
            name="gpt2_tts",
            pat_str=gpt2._pat_str,
            mergeable_ranks=gpt2._mergeable_ranks,
            special_tokens=special_tokens,
        )
        self.pad_id         = self.text_tokenizer.encode_single_token(self.EOS_TOKEN)
        self.in_eos_id      = self.text_tokenizer.encode_single_token(self.EOS_TOKEN)
        self.bos_id         = self.text_tokenizer.encode_single_token(self.BOS_TOKEN)
        self.audio_start_id = self.text_tokenizer.encode_single_token(self.AUDIO_START_TOKEN)

        self.audio_vocab_size = self.wav_tokenizer.feature_extractor.encodec.quantizer.bins
        self.text_vocab_size = self.text_tokenizer.n_vocab
        self.audio_offset = self.text_vocab_size
        # @psando: pad in_vocab_size up to a multiple of 64 for GPU efficiency
        #          TODO: out_vocab_size should be padded too, but requires model.py generate changes
        self.in_vocab_size = math.ceil((self.text_vocab_size + self.audio_vocab_size) / 64) * 64
        self.out_vocab_size = self.audio_vocab_size + 1 # @psando: +1 for EOS
        self.out_eos_id = self.audio_vocab_size         # @psando: last output id

        assert self.audio_vocab_size > 0

    def encode_text(self, text):
        return self.text_tokenizer.encode(text, allowed_special="all") # @psando: allowed_special='all' important if we include special token NV tags

    def encode_audio(self, waveform):
        _, audio_ids = self.wav_tokenizer.encode_infer(
            waveform,
            bandwidth_id=self.wav_tokenizer.bandwidth_id,
        )
        raw_audio_ids = audio_ids.reshape(-1).tolist()
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

        if audio_ids.numel() == 0:
            return None

        # generated sequence is in input space, so convert audio tokens back to output space
        audio_ids = audio_ids - self.audio_offset
        features = self.wav_tokenizer.codes_to_features(audio_ids)
        return self.wav_tokenizer.decode(
            features,
            bandwidth_id=self.wav_tokenizer.bandwidth_id,
        )


def create_joint_tokenizer(wav_tokenizer):
    return JointTokenizer(wav_tokenizer=wav_tokenizer)
