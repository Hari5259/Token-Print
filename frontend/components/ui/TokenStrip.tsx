"use client";

import { useStore } from "@/lib/store";
import { specTokenLabel } from "@/lib/specDecoding";
import { tokenClass } from "@/lib/tokenDisplay";

function disp(t: string): string {
  const s = t.replace(/\n/g, "⏎");
  return s.length === 0 ? "␣" : s;
}

export default function TokenStrip() {
  const meta = useStore((s) => s.genMeta);
  const frames = useStore((s) => s.genFrames);
  const status = useStore((s) => s.genStatus);
  if (!meta) return null;

  const gen = frames.filter((f) => !f.eos);
  // Show only the tail of the (often long chat-template) prompt so the
  // generated tokens stay prominent.
  const TAIL = 8;
  const promptTail = meta.prompt_tokens.slice(-TAIL);
  const truncated = meta.prompt_tokens.length > TAIL;

  return (
    <div className="token-strip">
      <span className="ts-label">tokens</span>
      {truncated && <span className="tok prompt">…</span>}
      {promptTail.map((t, i) => (
        <span key={"p" + i} className="tok prompt">
          {disp(t)}
        </span>
      ))}
      {gen.map((f, i) => (
        <span
          key={"g" + i}
          className={tokenClass(f, i === gen.length - 1)}
          title={specTokenLabel(f) || undefined}
        >
          {disp(f.chosen.text)}
        </span>
      ))}
      {status === "streaming" && <span className="tok cursor">▌</span>}
      {gen.length > 0 && gen[0].accepted_drafts && (
        <span className="ts-legend" style={{ marginLeft: "8px", fontSize: "0.7em", opacity: 0.6 }}>
          (orange: accepted draft, gold: target continuation; first token is from prefill)
        </span>
      )}
    </div>
  );
}
