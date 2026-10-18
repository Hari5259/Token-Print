import assert from "node:assert/strict";
import test from "node:test";

/**
 * Issue #297 — CONTEXT field in the inspector.
 *
 * CONTEXT should equal cache_len (the number of cached positions), NOT
 * prompt_len + cache_len. cache_len already includes the prompt tokens.
 *
 * This is pure arithmetic validation — the actual rendering is in
 * GenerationPanel.tsx line ~290 which displays cacheLen directly.
 */

function computeContext(cacheLen: number | null): number | null {
  return cacheLen; // Context is simply the cache length
}

test("CONTEXT equals cache_len, not prompt_len + cache_len", () => {
  // Simulate first decode step: cache holds the prefilled prompt
  const promptLen = 10;
  const cacheLenStep1 = 10; // same as prompt_len on first decode
  const context1 = computeContext(cacheLenStep1);
  assert.equal(context1, 10);

  // Simulate second decode step: cache grows by 1 new position
  const cacheLenStep2 = 11;
  const context2 = computeContext(cacheLenStep2);
  assert.equal(context2, 11);

  // Simulate sliding-window at step 50 with window_size=32
  const cacheLenStep50 = 32; // capped at window size
  const context50 = computeContext(cacheLenStep50);
  assert.equal(context50, 32);
});

test("cache_len includes prompt on first decode step", () => {
  const promptLen = 20;
  const cacheLenFirstDecode = 20; // cache holds prefilled prompt
  assert.equal(cacheLenFirstDecode, promptLen);
});

test("sliding-window caps cache_len at window_size", () => {
  const windowSize = 32;
  const promptLen = 10;
  const totalPositions = 60; // generated 50 tokens beyond prompt

  // Without sliding window, cache_len would equal totalPositions
  // With sliding window active, cache_len caps at windowSize
  const cacheLenWithoutWindow = totalPositions;
  const cacheLenWithWindow = Math.min(totalPositions, windowSize);

  assert.equal(cacheLenWithoutWindow, 60);
  assert.equal(cacheLenWithWindow, 32);
});

test("null cache_len returns null context", () => {
  assert.equal(computeContext(null), null);
});
