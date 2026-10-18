"use client";

import { useMemo, useRef } from "react";
import { Billboard, Html, Line, Text } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { Color, Mesh, MeshStandardMaterial } from "three";

import { useStore } from "@/lib/store";

const STACK_X = -8;
const SLAB_GAP = 0.5;
const BAR_SCALE = 8; // world height for probability 1.0

// The probability skyline is laid out around x = 4 (see the skyline below).
const SKYLINE_X = 4;

function layerColor(i: number, n: number): Color {
  // blue at the input (layer 0) -> warm near the output.
  return new Color().setHSL(0.6 - (i / Math.max(1, n - 1)) * 0.42, 0.8, 0.5);
}

/**
 * Speculative-decoding readout (issue #297).
 *
 * Shows the current token's speculative status as a fixed 2D overlay.
 * Everything drawn here is a real frame value: nothing is hardcoded.
 */
function SpeculativeReadout({
  nAccepted,
  draftPos,
  gamma,
}: {
  nAccepted: number;
  draftPos: number;
  gamma: number | undefined;
}) {
  const isContinuation = draftPos >= nAccepted;
  const label = isContinuation
    ? `spec: ${nAccepted}/${gamma ?? "?"} drafts accepted`
    : `spec: draft ${draftPos + 1} of ${nAccepted} accepted`;

  return (
    <Html fullscreen style={{ pointerEvents: "none" }}>
      <div
        style={{
          position: "absolute",
          top: "20px",
          right: "20px",
          fontFamily: "monospace",
          fontSize: "14px",
          fontWeight: 600,
          color: isContinuation ? "#ffd36e" : "#ff9f43",
          background: "rgba(0,0,0,0.7)",
          padding: "8px 12px",
          borderRadius: "4px",
          border: `1px solid ${isContinuation ? "#ffd36e" : "#ff9f43"}`,
          textShadow: "0 0 4px rgba(0,0,0,0.8)",
          zIndex: 100,
        }}
      >
        {label}
      </div>
    </Html>
  );
}

/**
 * Sliding-window KV cache visual (issue #297).
 *
 * Rendered as a fixed 2D overlay in the viewport (not world space) so it's
 * always visible regardless of camera position. Shows the real cache status:
 * kept vs trimmed positions, boundary line, and numeric readout. All values
 * from real frame fields and meta.
 *
 * Source: model.py:1232-1234 (trim_cache call and cache_len_in = window_size).
 */
function SlidingWindowBar({
  cacheLen,
  windowSize,
  totalPositions,
}: {
  cacheLen: number;
  windowSize: number;
  totalPositions: number;
}) {
  if (totalPositions <= 0) return null;

  const isTrimming = cacheLen >= windowSize;
  const windowFrac = Math.min(cacheLen / Math.max(totalPositions, 1), 1.0);

  return (
    <Html fullscreen style={{ pointerEvents: "none" }}>
      <div
        style={{
          position: "absolute",
          bottom: "80px",
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "4px",
          zIndex: 100,
        }}
      >
        {/* Bar container */}
        <div
          style={{
            width: "400px",
            height: "16px",
            background: "#1a2030",
            borderRadius: "3px",
            position: "relative",
            overflow: "visible",
            border: "1px solid rgba(255,255,255,0.1)",
          }}
        >
          {/* Active window overlay */}
          <div
            style={{
              position: "absolute",
              right: 0,
              top: 0,
              height: "100%",
              width: `${windowFrac * 100}%`,
              background: isTrimming
                ? "linear-gradient(90deg, rgba(255,211,110,0.3), rgba(255,211,110,0.5))"
                : "linear-gradient(90deg, rgba(87,208,255,0.3), rgba(87,208,255,0.5))",
              borderRadius: "2px",
            }}
          />
          {/* Boundary line at window limit */}
          {isTrimming && (
            <div
              style={{
                position: "absolute",
                right: `${(1 - windowFrac) * 100}%`,
                top: "-2px",
                bottom: "-2px",
                width: "2px",
                background: "#ff6b6b",
              }}
            />
          )}
        </div>
        {/* Label */}
        <div
          style={{
            fontFamily: "monospace",
            fontSize: "11px",
            color: isTrimming ? "#ffd36e" : "#57d0ff",
            whiteSpace: "nowrap",
          }}
        >
          {isTrimming
            ? `KV window: ${cacheLen} / ${windowSize} (trimming oldest)`
            : `KV cache: ${cacheLen} / ${totalPositions} positions`}
        </div>
      </div>
    </Html>
  );
}

/**
 * Needle recall banner (issue #297).
 * needle_report only exists on the done frame — no per-token attention field
 * is emitted. We show the recall result as a billboard once generation ends.
 * Source: model.py:1418-1425 (needle_report built post-loop, in done dict).
 */
function NeedleRecallBanner({
  needle,
  recalled,
}: {
  needle: string;
  recalled: boolean;
}) {
  const short = needle.length > 28 ? needle.slice(0, 25) + "…" : needle;
  return (
    <Billboard position={[SKYLINE_X, BAR_SCALE + 3.0, 0]}>
      <Text
        fontSize={0.38}
        color={recalled ? "#55efc4" : "#ff6b6b"}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {recalled ? `✓ needle recalled: "${short}"` : `✗ needle missed: "${short}"`}
      </Text>
    </Billboard>
  );
}

/**
 * Generation District: a vertical stack of transformer-layer slabs whose glow
 * is driven by the REAL per-layer mean activation magnitude of the current
 * generation step, plus a probability skyline of the real top-k candidates.
 * Everything is driven by recorded stream frames (replayable), never simulated.
 */
export default function TransformerDistrict() {
  const meta = useStore((s) => s.genMeta);
  const frames = useStore((s) => s.genFrames);
  const playIndex = useStore((s) => s.playIndex);
  const genStatus = useStore((s) => s.genStatus);
  const genDone = useStore((s) => s.genDone);

  const nLayers = meta?.num_layer_stats ?? 25;
  const topK = meta?.top_k ?? 10;

  const frame =
    playIndex >= 0 && playIndex < frames.length
      ? frames[playIndex]
      : frames.length
        ? frames[frames.length - 1]
        : null;

  // Does ANY frame in this run carry speculative-decoding annotations? Drives
  // whether the readout is mounted at all, so greedy runs pay nothing.
  const hasSpecData = useMemo(
    () => frames.some((f) => f.accepted_drafts === true),
    [frames],
  );

  // Speculative fields for the CURRENT frame (absent on greedy frames).
  const specNAccepted = frame?.n_accepted;
  const specDraftPos = frame?.draft_pos;
  const specGamma = meta?.decoding_params?.draft_gamma;

  // Sliding-window fields — cache_len is the real fed-cache length per frame.
  const isSlidingWindow = meta?.decoding === "sliding_window";
  const windowSize = meta?.decoding_params?.window_size ?? 512;
  const cacheLen = frame?.cache_len ?? 0;
  // total positions seen so far = prompt_len + step index (approx from frame.step)
  const totalPositions = (meta?.prompt_len ?? 0) + (frame?.step ?? 0) + 1;

  // Needle recall — only on done frame, shown after generation finishes.
  const needleReport = genDone?.needle_report ?? null;

  const slabRefs = useRef<Array<Mesh | null>>([]);
  const barRefs = useRef<Array<Mesh | null>>([]);

  const stackHeight = (nLayers - 1) * SLAB_GAP;

  useFrame(() => {
    const stats = frame?.layer_stats ?? [];
    const max = stats.length ? Math.max(...stats) : 1;

    // Layer slabs: emissive brightness ~ normalized activation at that layer.
    slabRefs.current.forEach((m, i) => {
      if (!m) return;
      const norm = max > 0 ? (stats[i] ?? 0) / max : 0;
      const mat = m.material as MeshStandardMaterial;
      const target = 0.08 + norm * 2.4;
      mat.emissiveIntensity += (target - mat.emissiveIntensity) * 0.15;
    });

    // Skyline bars: height ~ probability (unit-height box scaled on Y).
    const tk = frame?.topk ?? [];
    barRefs.current.forEach((m, k) => {
      if (!m) return;
      const target = (tk[k]?.prob ?? 0) * BAR_SCALE + 0.02;
      m.scale.y += (target - m.scale.y) * 0.2;
      m.position.y = m.scale.y / 2;
    });
  });

  const slabs = useMemo(
    () => Array.from({ length: nLayers }, (_, i) => layerColor(i, nLayers)),
    [nLayers],
  );

  const tk = frame?.topk ?? [];

  return (
    <group>
      {/* --- Transformer layer stack --- */}
      <Billboard position={[STACK_X, stackHeight + 1.6, 0]}>
        <Text fontSize={0.5} anchorX="center" color="#e6ecff" outlineWidth={0.02} outlineColor="#000000">
          Layers (activation magnitude)
        </Text>
      </Billboard>
      {slabs.map((c, i) => (
        <mesh
          key={i}
          ref={(el) => void (slabRefs.current[i] = el)}
          position={[STACK_X, i * SLAB_GAP, 0]}
        >
          <boxGeometry args={[5, 0.34, 5]} />
          <meshStandardMaterial
            color={c}
            emissive={c}
            emissiveIntensity={0.1}
            roughness={0.5}
            metalness={0.2}
          />
        </mesh>
      ))}
      <Billboard position={[STACK_X, -1.1, 0]}>
        <Text fontSize={0.3} anchorX="center" color="#8a97bd">
          layer 0 (embeddings) → layer {nLayers - 1}
        </Text>
      </Billboard>

      {/* --- Probability skyline (real top-k) --- */}
      <Billboard position={[4, 9.4, 0]}>
        <Text fontSize={0.5} anchorX="center" color="#e6ecff" outlineWidth={0.02} outlineColor="#000000">
          Next-token probabilities (top {topK})
        </Text>
      </Billboard>
      {Array.from({ length: topK }, (_, k) => {
        const x = 4 + (k - (topK - 1) / 2) * 1.5;
        const chosen = k === 0; // greedy: argmax is topk[0]
        const col = chosen ? "#ffd36e" : "#57d0ff";
        const cand = tk[k];
        return (
          <group key={k} position={[x, 0, 0]}>
            <mesh ref={(el) => void (barRefs.current[k] = el)} position={[0, 0, 0]}>
              <boxGeometry args={[1.05, 1, 1.05]} />
              <meshStandardMaterial
                color={col}
                emissive={col}
                emissiveIntensity={chosen ? 0.85 : 0.35}
                roughness={0.4}
                metalness={0.1}
              />
            </mesh>
            <Billboard position={[0, 8.9, 0]}>
              <Text
                fontSize={0.34}
                anchorX="center"
                color={chosen ? "#ffe6a0" : "#cfe8ff"}
                outlineWidth={0.015}
                outlineColor="#000000"
              >
                {cand ? cand.text.trim() || "␣" : ""}
              </Text>
              <Text position={[0, -0.42, 0]} fontSize={0.26} anchorX="center" color="#8a97bd">
                {cand ? `${(cand.prob * 100).toFixed(0)}%` : ""}
              </Text>
            </Billboard>
          </group>
        );
      })}

      {/* --- Speculative decoding readout (issue #297) --- */}
      {hasSpecData &&
        frame?.accepted_drafts === true &&
        specNAccepted != null &&
        specDraftPos != null && (
          <SpeculativeReadout
            nAccepted={specNAccepted}
            draftPos={specDraftPos}
            gamma={specGamma}
          />
        )}

      {/* --- Sliding-window KV cache bar (issue #297) --- */}
      {isSlidingWindow && cacheLen > 0 && (
        <SlidingWindowBar
          cacheLen={cacheLen}
          windowSize={windowSize}
          totalPositions={totalPositions}
        />
      )}

      {/* --- Needle recall banner (issue #297) — post-generation only --- */}
      {needleReport && (
        <NeedleRecallBanner
          needle={needleReport.needle}
          recalled={needleReport.recalled}
        />
      )}

      {/* --- Generated text / status readout --- */}
      <Billboard position={[0, -3.4, 0]}>
        <Text
          fontSize={0.5}
          maxWidth={34}
          textAlign="center"
          anchorX="center"
          color="#b6f0c8"
          outlineWidth={0.015}
          outlineColor="#000000"
        >
          {frames.length === 0
            ? genStatus === "streaming"
              ? "generating…"
              : "Enter a prompt and press Generate."
            : `“${frames
                .slice(0, (frame?.step ?? frames.length - 1) + 1)
                .filter((f) => !f.eos)
                .map((f) => f.chosen.text)
                .join("")
                .trim()}”`}
        </Text>
      </Billboard>
    </group>
  );
}
