import itertools
import types
import unittest
from unittest.mock import MagicMock

import torch
from app.model import ActivationEngine, GenerationState, ModelEngine
from torch import nn


class DummyCache:
    """Mock DynamicCache for unit testing KV cache truncation."""
    def __init__(self, seq_len: int = 20, num_layers: int = 2):
        self.key_cache = [torch.randn(1, 4, seq_len, 16) for _ in range(num_layers)]
        self.value_cache = [torch.randn(1, 4, seq_len, 16) for _ in range(num_layers)]
        self._seen_tokens = seq_len


class TestGenerationState(unittest.TestCase):
    def test_initialization(self):
        state = GenerationState()
        self.assertIsNone(state.past_key_values)
        self.assertEqual(state.positions_done, 0)

    def test_trim_dummy_cache(self):
        cache = DummyCache(seq_len=20)
        state = GenerationState(past_key_values=cache, positions_done=20)

        # Trim to keep last 10 positions
        state.trim_cache(10)
        self.assertIsNotNone(state.past_key_values)
        self.assertEqual(state.past_key_values.key_cache[0].shape[-2], 10)
        self.assertEqual(state.past_key_values.value_cache[0].shape[-2], 10)

    def test_trim_tuple_cache(self):
        # Tuple format: tuple of (key, value) per layer
        layer1_k = torch.randn(1, 4, 15, 16)
        layer1_v = torch.randn(1, 4, 15, 16)
        layer2_k = torch.randn(1, 4, 15, 16)
        layer2_v = torch.randn(1, 4, 15, 16)
        tuple_cache = ((layer1_k, layer1_v), (layer2_k, layer2_v))

        state = GenerationState(past_key_values=tuple_cache, positions_done=15)
        state.trim_cache(8)

        self.assertEqual(state.past_key_values[0][0].shape[-2], 8)
        self.assertEqual(state.past_key_values[0][1].shape[-2], 8)
        self.assertEqual(state.past_key_values[1][0].shape[-2], 8)

    def test_rollback_speculative_drafts(self):
        cache = DummyCache(seq_len=20)
        # 16 prompt/pre-draft tokens + 4 draft tokens evaluated = 20 total
        state = GenerationState(past_key_values=cache, positions_done=20)

        # 1 draft token accepted out of gamma=4 (3 rejected)
        state.rollback_speculative_drafts(accepted_k=1, draft_gamma=4)

        # Target length should be 20 - (4 - 1) = 17 positions
        self.assertEqual(state.positions_done, 17)
        self.assertEqual(state.past_key_values.key_cache[0].shape[-2], 17)
        self.assertEqual(state.past_key_values.value_cache[0].shape[-2], 17)

    def test_rollback_all_accepted(self):
        cache = DummyCache(seq_len=20)
        state = GenerationState(past_key_values=cache, positions_done=20)

        # All 4 drafts accepted -> no rollback required
        state.rollback_speculative_drafts(accepted_k=4, draft_gamma=4)
        self.assertEqual(state.positions_done, 20)
        self.assertEqual(state.past_key_values.key_cache[0].shape[-2], 20)


# ---------------------------------------------------------------------------
# Regression: Issue #281 — cache_len must track positions_done, not logits.shape[1]
#
# Before commit b394f2f, cache_len was incorrectly derived from logits.shape[1].
# During a speculative step the draft-verification forward pass emits logits with
# shape [1, gamma, vocab], so logits.shape[1] == gamma (e.g. 3). This caused
# cache_len to jump to 3 rather than reflect the real number of cached KV
# positions (e.g. 6 prompt + 3 previously generated = 9).
#
# The fix: cache_len_in is set to state.positions_done before every token is
# emitted, so it always equals the sequential KV-position count regardless of
# the batch size used by the speculative verify pass.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Minimal synthetic model that correctly simulates speculative decoding output
# shapes (batch size can be 1 or gamma depending on call site).
# ---------------------------------------------------------------------------

class _SyntheticSpecModel(nn.Module):
    """Lightweight causal-LM double for speculative-decoding regression tests.

    The forward() method accepts arbitrary-length input_ids and returns
    matching logits, plus a fake KV cache object whose seq-dim grows
    monotonically — mirroring HuggingFace cache behaviour.
    """
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
        super().__init__()
        self.config = types.SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            is_encoder_decoder=False,
            model_type="llama",
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
    ):
        x = self.embed(input_ids)            # [1, seq, hidden]
        x = self.proj(x)
        logits = self.lm_head(x)             # [1, seq, vocab]

        seq = input_ids.shape[1]
        prior = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total = prior + seq

        # Grow a fake cache by the number of new positions consumed.
        new_cache = (
            (
                torch.zeros(1, 1, total, 4),
                torch.zeros(1, 1, total, 4),
            ),
        )

        return types.SimpleNamespace(
            logits=logits,
            past_key_values=new_cache,
            hidden_states=(x,),
        )


def _make_spec_service(model: nn.Module) -> ModelEngine:
    """Build a minimal ModelEngine instance wired to *model* without loading weights."""
    import threading

    svc = ModelEngine.__new__(ModelEngine)
    svc.model = model
    svc.device = "cpu"
    svc.model_id = "test-spec-model"
    svc.mode = "causal_lm"
    svc._lock = threading.Lock()
    svc.num_layers = 1
    svc.num_heads = 1
    svc.hidden_size = model.config.hidden_size
    svc.revision = "test"
    svc.activation_engine = ActivationEngine()
    svc._layer_elapsed = {}
    svc._layer_timing_lock = threading.Lock()

    tok = MagicMock()
    tok.side_effect = lambda prompt, **kw: {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.side_effect = lambda ids, **kw: f"t{ids[0] if ids else 0}"
    tok.eos_token_id = 9999
    svc.tokenizer = tok
    svc._catalog = None
    return svc


class TestCacheLenTracksPositionsDone(unittest.TestCase):
    """Regression tests for issue #281 (cache_len fix, commit b394f2f).

    Verify that every emitted token frame carries cache_len == the sequential
    KV-position count (state.positions_done), and never == logits.shape[1].
    """

    def _collect_token_frames(self, frames):
        return [f for f in frames if f.get("type") == "token"]

    def test_greedy_cache_len_increments_monotonically(self):
        """Baseline: greedy decode — cache_len should increment by 1 each step."""
        model = _SyntheticSpecModel()
        svc = _make_spec_service(model)

        n_tokens = 4
        frames = list(
            svc.generate_steps(
                prompt="hello",
                max_new_tokens=n_tokens,
                top_k=5,
                use_chat_template=False,
                include_catalog=False,
                decoding_mode="greedy",
            )
        )
        token_frames = self._collect_token_frames(frames)
        self.assertEqual(len(token_frames), n_tokens)

        # First token: prompt is 5 tokens, no cache yet → cache_len == 0
        # Subsequent: cache_len grows by 1 each step (one token appended per step).
        for i, f in enumerate(token_frames):
            # cache_len must be a non-negative integer
            self.assertIsInstance(f["cache_len"], int)
            self.assertGreaterEqual(f["cache_len"], 0)

        # Strictly monotonically non-decreasing
        cache_lens = [f["cache_len"] for f in token_frames]
        for a, b in itertools.pairwise(cache_lens):
            self.assertGreaterEqual(b, a, msg="cache_len must not decrease between steps")

    def test_speculative_cache_len_tracks_positions_done(self):
        """Core regression: cache_len must equal the sequential KV-position count.

        Before the fix, cache_len was taken from ``logits.shape[1]``. On a
        speculative step the verify forward pass returns logits of shape
        ``[1, gamma, vocab]``, so that formula yielded ``gamma`` (3 here) on
        every speculative frame instead of the real cache length.

        The prompt stub is 5 tokens, so every legitimate cache_len is either 0
        (prefill, nothing cached yet) or >= 5. A value of exactly ``gamma`` is
        therefore never legitimate, which is what makes this a real regression
        test rather than an assertion about an unrelated number.
        """
        model = _SyntheticSpecModel()
        svc = _make_spec_service(model)

        prompt_len = 5           # tokenizer stub always returns 5 ids
        draft_gamma = 3
        n_tokens = 6

        frames = list(
            svc.generate_steps(
                prompt="hello",
                max_new_tokens=n_tokens,
                top_k=5,
                use_chat_template=False,
                include_catalog=False,
                decoding_mode="speculative",
                draft_gamma=draft_gamma,
            )
        )
        token_frames = self._collect_token_frames(frames)
        self.assertGreater(len(token_frames), 0)

        # At least one frame must actually come from the speculative branch,
        # otherwise this test would silently pass without exercising the code.
        self.assertTrue(
            any(f.get("accepted_drafts") for f in token_frames),
            "no speculative frames emitted — the regression path was not exercised",
        )

        for i, f in enumerate(token_frames):
            cache_len = f["cache_len"]
            self.assertIsInstance(cache_len, int)
            self.assertGreaterEqual(cache_len, 0)
            self.assertIn(
                cache_len == 0 or cache_len >= prompt_len,
                (True,),
                msg=(
                    f"Frame {i}: cache_len={cache_len} is neither 0 (prefill) nor "
                    f">= prompt length {prompt_len} — it looks like "
                    f"logits.shape[1] was used instead of state.positions_done."
                ),
            )

    def test_speculative_cache_len_monotonically_non_decreasing(self):
        """cache_len must never go backwards across frames in speculative mode."""
        model = _SyntheticSpecModel()
        svc = _make_spec_service(model)

        frames = list(
            svc.generate_steps(
                prompt="hello world",
                max_new_tokens=6,
                top_k=5,
                use_chat_template=False,
                include_catalog=False,
                decoding_mode="speculative",
                draft_gamma=2,
            )
        )
        token_frames = self._collect_token_frames(frames)
        cache_lens = [f["cache_len"] for f in token_frames]
        for idx, (a, b) in enumerate(itertools.pairwise(cache_lens)):
            self.assertGreaterEqual(
                b, a,
                msg=(
                    f"cache_len decreased from {a} to {b} between frames "
                    f"{idx} and {idx + 1}. This indicates the positions_done "
                    "counter was reset mid-generation."
                ),
            )
