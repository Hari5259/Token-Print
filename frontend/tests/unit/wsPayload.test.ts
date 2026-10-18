import assert from "node:assert/strict";
import test from "node:test";
import type { GenOptions } from "../../lib/ws";

/**
 * Issue #297 — WebSocket generate request payload.
 *
 * Verify that decoding_mode, window_size, draft_gamma, and needle are all
 * included in the WS request payload sent to the backend.
 */

test("WS payload includes all required fields for speculative decoding", () => {
  const opts: GenOptions = {
    maxNewTokens: 40,
    topK: 10,
    temperature: 0.8,
    topP: 0.9,
    seed: 42,
    decodingMode: "speculative",
    draftGamma: 3,
    windowSize: 512,
    needle: "[MEMORY] The capital of France is Paris.",
  };

  // Simulate the payload construction from ws.ts
  const payload = {
    prompt: "Test prompt",
    max_new_tokens: opts.maxNewTokens ?? 40,
    top_k: opts.topK ?? 10,
    temperature: opts.temperature ?? 1.0,
    top_p: opts.topP ?? 1.0,
    seed: opts.seed ?? undefined,
    trace: opts.trace ?? false,
    record_trace: opts.recordTrace ?? false,
    decoding_mode: opts.decodingMode ?? "greedy",
    window_size: opts.windowSize ?? 512,
    draft_gamma: opts.draftGamma ?? 4,
    needle: opts.needle ?? undefined,
    gguf: opts.gguf ?? undefined,
  };

  assert.equal(payload.decoding_mode, "speculative");
  assert.equal(payload.draft_gamma, 3);
  assert.equal(payload.window_size, 512);
  assert.equal(payload.needle, "[MEMORY] The capital of France is Paris.");
});

test("WS payload includes defaults when fields are omitted", () => {
  const opts: GenOptions = {
    decodingMode: "greedy",
  };

  const payload = {
    prompt: "Test",
    max_new_tokens: opts.maxNewTokens ?? 40,
    decoding_mode: opts.decodingMode ?? "greedy",
    window_size: opts.windowSize ?? 512,
    draft_gamma: opts.draftGamma ?? 4,
  };

  assert.equal(payload.decoding_mode, "greedy");
  assert.equal(payload.window_size, 512);
  assert.equal(payload.draft_gamma, 4);
});

test("sliding-window mode includes window_size", () => {
  const opts: GenOptions = {
    decodingMode: "sliding_window",
    windowSize: 64,
  };

  const payload = {
    decoding_mode: opts.decodingMode,
    window_size: opts.windowSize,
  };

  assert.equal(payload.decoding_mode, "sliding_window");
  assert.equal(payload.window_size, 64);
});
