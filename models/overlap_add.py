"""
Windowed streaming inference with crossfaded overlap-add reconstruction.

Splits a long token sequence into overlapping windows, runs the editor on
each window independently, then merges output tokens using crossfaded
overlap-add to avoid click artifacts at boundaries.
"""

import torch
import numpy as np


class StreamingOverlapAdd:
    """
    Process a long token sequence through the editor in overlapping windows
    and reconstruct a continuous output via overlap-add with cosine crossfades.
    """

    def __init__(self, window_tokens: int, hop_tokens: int,
                 crossfade_tokens: int = 4):
        """
        Args:
            window_tokens: Number of tokens per processing window.
            hop_tokens: Number of tokens to advance between windows.
            crossfade_tokens: Tokens over which to crossfade in overlapping regions.
        """
        assert hop_tokens <= window_tokens, "Hop must be <= window"
        self.window_tokens = window_tokens
        self.hop_tokens = hop_tokens
        self.crossfade_tokens = crossfade_tokens

    def _cosine_fade(self, length: int) -> np.ndarray:
        """Half-cosine fade curve from 0 to 1."""
        return 0.5 * (1 - np.cos(np.pi * np.arange(length) / length))

    def segment(self, tokens: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """
        Split a 1-D token array into overlapping windows.

        Returns:
            List of (start_index, token_window) tuples.
        """
        T = len(tokens)
        windows = []
        start = 0
        while start < T:
            end = min(start + self.window_tokens, T)
            windows.append((start, tokens[start:end]))
            if end == T:
                break
            start += self.hop_tokens
        return windows

    def merge(self, windows: list[tuple[int, np.ndarray]],
              total_length: int) -> np.ndarray:
        """
        Overlap-add merge edited token windows back into a single sequence.

        For tokens (discrete), we operate on the underlying continuous
        embeddings/logits. For simplicity this merges by weighted selection:
        in overlapping regions, use crossfade weights to blend by selecting
        the token from the window with higher weight at each position.

        Args:
            windows: List of (start_index, edited_tokens) from editor.
            total_length: Expected output length.

        Returns:
            Merged token array of shape (total_length,).
        """
        output = np.full(total_length, -1, dtype=np.int64)
        weight = np.zeros(total_length, dtype=np.float32)

        for start, tokens in windows:
            end = start + len(tokens)
            w = np.ones(len(tokens), dtype=np.float32)

            # Fade-in at the start (except for the very first window)
            if start > 0:
                fade_len = min(self.crossfade_tokens, len(tokens))
                w[:fade_len] *= self._cosine_fade(fade_len)

            # Fade-out at the end (except for the very last window)
            if end < total_length:
                fade_len = min(self.crossfade_tokens, len(tokens))
                w[-fade_len:] *= self._cosine_fade(fade_len)[::-1]

            # Weighted selection: keep token from higher-weight window
            for i, (tok, wi) in enumerate(zip(tokens, w)):
                pos = start + i
                if pos < total_length and wi > weight[pos]:
                    output[pos] = tok
                    weight[pos] = wi

        # Fill any remaining gaps (shouldn't happen with proper overlap)
        unfilled = output == -1
        if unfilled.any():
            # Forward-fill
            for i in range(1, total_length):
                if output[i] == -1 and output[i - 1] != -1:
                    output[i] = output[i - 1]

        return output


def overlap_add_waveform(windows: list[tuple[int, np.ndarray]],
                         total_samples: int,
                         crossfade_samples: int = 480) -> np.ndarray:
    """
    Overlap-add for decoded waveform segments with cosine crossfading.

    This is applied AFTER token-level OLA and WavTokenizer decoding,
    to smooth any residual discontinuities at window boundaries.

    Args:
        windows: List of (start_sample, waveform_segment).
        total_samples: Total output waveform length.
        crossfade_samples: Crossfade length in audio samples.

    Returns:
        Smoothly reconstructed waveform of shape (total_samples,).
    """
    output = np.zeros(total_samples, dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)

    fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(crossfade_samples)
                                  / crossfade_samples))
    fade_out = fade_in[::-1]

    for start, wav in windows:
        end = start + len(wav)
        w = np.ones(len(wav), dtype=np.float32)

        if start > 0:
            fl = min(crossfade_samples, len(wav))
            w[:fl] *= fade_in[:fl]
        if end < total_samples:
            fl = min(crossfade_samples, len(wav))
            w[-fl:] *= fade_out[-fl:]

        seg_end = min(end, total_samples)
        seg_len = seg_end - start
        output[start:seg_end] += wav[:seg_len] * w[:seg_len]
        weight[start:seg_end] += w[:seg_len]

    # Normalize by total weight
    nonzero = weight > 0
    output[nonzero] /= weight[nonzero]
    return output
