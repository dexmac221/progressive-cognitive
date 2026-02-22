import { useState } from "react";

const phases = [
  {
    name: "Fase 1 — Fondamenta",
    samples: 50000,
    epochs: 8,
    seqLen: 256,
    minPerEpoch: 4.5,
    desc: "Calcolo esatto, tutti i pesi LoRA attivi",
  },
  {
    name: "Fase 2 — Consolidamento",
    samples: 30000,
    epochs: 6,
    seqLen: 256,
    minPerEpoch: 3.0,
    desc: "Pruning 30% + fine-tune intuitivo",
  },
  {
    name: "Fase 3 — Delega",
    samples: 30000,
    epochs: 6,
    seqLen: 256,
    minPerEpoch: 3.0,
    desc: "Routing + tool delegation",
  },
  {
    name: "Fase 4 — Orchestrazione",
    samples: 20000,
    epochs: 4,
    seqLen: 256,
    minPerEpoch: 2.0,
    desc: "Pipeline completa: intuizione → tool → validazione",
  },
];

export default function TrainingEstimator() {
  const [gpuCount, setGpuCount] = useState(2);
  const [batchSize, setBatchSize] = useState(32);
  const [loraRank, setLoraRank] = useState(16);
  const [useQuantization, setUseQuantization] = useState(false);

  const speedMultiplier = gpuCount === 2 ? 1.7 : 1.0; // ~85% scaling
  const batchEffect = batchSize >= 32 ? 1.0 : batchSize >= 16 ? 1.3 : 1.8;
  const quantEffect = useQuantization ? 0.75 : 1.0;
  const rankEffect = loraRank <= 8 ? 0.85 : loraRank <= 16 ? 1.0 : 1.2;

  const trainableParams = Math.round(
    (loraRank / 16) * 2.8 * (useQuantization ? 1 : 1)
  );

  const results = phases.map((p) => {
    const baseTime = p.epochs * p.minPerEpoch;
    const adjusted =
      (baseTime * batchEffect * quantEffect * rankEffect) / speedMultiplier;
    return { ...p, totalMin: adjusted };
  });

  const totalMin = results.reduce((s, r) => s + r.totalMin, 0);
  const totalHours = totalMin / 60;

  const vramPerGpu = useQuantization ? 8 : 14;
  const modelSize = useQuantization ? 0.6 : 2.2;

  const mono = "'JetBrains Mono', 'Fira Code', monospace";
  const serif = "'Instrument Serif', Georgia, serif";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0b0d",
        color: "#e0e0e0",
        fontFamily: serif,
        padding: "32px 20px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500&display=swap"
        rel="stylesheet"
      />

      <div style={{ maxWidth: 680, width: "100%" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <h1
            style={{
              fontSize: 28,
              fontWeight: 400,
              margin: 0,
              letterSpacing: "-0.01em",
            }}
          >
            Training Estimator
          </h1>
          <p
            style={{
              fontFamily: mono,
              fontSize: 11,
              color: "rgba(255,255,255,0.3)",
              marginTop: 8,
              letterSpacing: "0.08em",
            }}
          >
            TINYLLAMA 1.1B + LORA • ARCHITETTURA COGNITIVA PROGRESSIVA
          </p>
        </div>

        {/* GPU Card */}
        <div
          style={{
            background: "linear-gradient(135deg, #0d1a0d 0%, #0a150f 100%)",
            border: "1px solid rgba(118,185,71,0.15)",
            borderRadius: 14,
            padding: 24,
            marginBottom: 20,
          }}
        >
          <div
            style={{
              fontFamily: mono,
              fontSize: 10,
              color: "#76b947",
              letterSpacing: "0.12em",
              marginBottom: 12,
            }}
          >
            HARDWARE
          </div>
          <div
            style={{
              fontSize: 20,
              marginBottom: 4,
            }}
          >
            {gpuCount}× NVIDIA RTX 6000 Ada Generation
          </div>
          <div
            style={{
              fontFamily: mono,
              fontSize: 12,
              color: "rgba(255,255,255,0.4)",
              lineHeight: 1.8,
            }}
          >
            {gpuCount * 48}GB VRAM totale • {gpuCount * 91.1} TFLOPS FP32 •{" "}
            {gpuCount * 18176} CUDA cores • {gpuCount * 568} Tensor Cores
          </div>
        </div>

        {/* Config */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            marginBottom: 24,
          }}
        >
          {[
            {
              label: "GPU",
              value: gpuCount,
              options: [
                [1, "1× RTX 6000"],
                [2, "2× RTX 6000"],
              ],
              set: setGpuCount,
            },
            {
              label: "Batch Size",
              value: batchSize,
              options: [
                [8, "8"],
                [16, "16"],
                [32, "32"],
              ],
              set: setBatchSize,
            },
            {
              label: "LoRA Rank",
              value: loraRank,
              options: [
                [8, "r=8 (~1.4M)"],
                [16, "r=16 (~2.8M)"],
                [32, "r=32 (~5.6M)"],
              ],
              set: setLoraRank,
            },
            {
              label: "Quantizzazione",
              value: useQuantization,
              options: [
                [false, "FP16"],
                [true, "4-bit QLoRA"],
              ],
              set: setUseQuantization,
            },
          ].map((cfg, i) => (
            <div
              key={i}
              style={{
                background: "rgba(255,255,255,0.02)",
                borderRadius: 10,
                padding: "14px 16px",
                border: "1px solid rgba(255,255,255,0.05)",
              }}
            >
              <div
                style={{
                  fontFamily: mono,
                  fontSize: 9,
                  color: "rgba(255,255,255,0.35)",
                  letterSpacing: "0.1em",
                  marginBottom: 8,
                }}
              >
                {cfg.label.toUpperCase()}
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                {cfg.options.map(([val, label]) => (
                  <button
                    key={String(val)}
                    onClick={() => cfg.set(val)}
                    style={{
                      flex: 1,
                      padding: "6px 4px",
                      background:
                        cfg.value === val
                          ? "rgba(118,185,71,0.15)"
                          : "rgba(255,255,255,0.03)",
                      border: `1px solid ${cfg.value === val ? "rgba(118,185,71,0.4)" : "rgba(255,255,255,0.06)"}`,
                      borderRadius: 6,
                      color:
                        cfg.value === val
                          ? "#76b947"
                          : "rgba(255,255,255,0.4)",
                      fontFamily: mono,
                      fontSize: 11,
                      cursor: "pointer",
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Risultato grande */}
        <div
          style={{
            background: "linear-gradient(135deg, #111318 0%, #0d1018 100%)",
            borderRadius: 14,
            padding: 28,
            marginBottom: 20,
            textAlign: "center",
            border: "1px solid rgba(107,141,214,0.12)",
          }}
        >
          <div
            style={{
              fontFamily: mono,
              fontSize: 10,
              color: "rgba(255,255,255,0.3)",
              letterSpacing: "0.12em",
              marginBottom: 12,
            }}
          >
            TEMPO STIMATO TOTALE
          </div>
          <div style={{ fontSize: 52, fontWeight: 400, color: "#6B8DD6" }}>
            {totalHours < 1
              ? `${Math.round(totalMin)} min`
              : `${totalHours.toFixed(1)}h`}
          </div>
          <div
            style={{
              fontFamily: mono,
              fontSize: 12,
              color: "rgba(255,255,255,0.35)",
              marginTop: 8,
            }}
          >
            {Math.round(totalMin)} minuti • 4 fasi •{" "}
            {(50 + 30 + 30 + 20) / 1000}K samples totali • VRAM:{" "}
            {vramPerGpu}GB/GPU
          </div>
        </div>

        {/* Breakdown fasi */}
        <div style={{ marginBottom: 24 }}>
          {results.map((r, i) => {
            const pct = (r.totalMin / totalMin) * 100;
            const colors = ["#E8654A", "#D4943A", "#5BA88C", "#6B8DD6"];
            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  padding: "12px 0",
                  borderBottom:
                    i < 3 ? "1px solid rgba(255,255,255,0.04)" : "none",
                }}
              >
                <div style={{ width: 140, flexShrink: 0 }}>
                  <div
                    style={{
                      fontFamily: mono,
                      fontSize: 11,
                      color: colors[i],
                    }}
                  >
                    Fase {i + 1}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: "rgba(255,255,255,0.35)",
                      fontFamily: mono,
                    }}
                  >
                    {r.samples / 1000}K × {r.epochs}ep
                  </div>
                </div>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      height: 8,
                      background: "rgba(255,255,255,0.04)",
                      borderRadius: 4,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${pct}%`,
                        background: colors[i],
                        borderRadius: 4,
                        opacity: 0.7,
                      }}
                    />
                  </div>
                </div>
                <div
                  style={{
                    width: 60,
                    textAlign: "right",
                    fontFamily: mono,
                    fontSize: 13,
                    color: "rgba(255,255,255,0.6)",
                  }}
                >
                  {r.totalMin < 60
                    ? `${Math.round(r.totalMin)}min`
                    : `${(r.totalMin / 60).toFixed(1)}h`}
                </div>
              </div>
            );
          })}
        </div>

        {/* Memory breakdown */}
        <div
          style={{
            background: "rgba(255,255,255,0.02)",
            borderRadius: 12,
            padding: 20,
            border: "1px solid rgba(255,255,255,0.04)",
            marginBottom: 20,
          }}
        >
          <div
            style={{
              fontFamily: mono,
              fontSize: 9,
              color: "rgba(255,255,255,0.3)",
              letterSpacing: "0.1em",
              marginBottom: 12,
            }}
          >
            MEMORY BUDGET PER GPU
          </div>
          <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
            {[
              {
                label: "Modello",
                gb: modelSize,
                color: "#6B8DD6",
              },
              {
                label: "LoRA",
                gb: trainableParams * 0.004,
                color: "#76b947",
              },
              {
                label: "Optimizer",
                gb: trainableParams * 0.016,
                color: "#D4943A",
              },
              {
                label: "Activations",
                gb: useQuantization ? 3 : 6,
                color: "#E8654A",
              },
              {
                label: "Libera",
                gb: 48 - vramPerGpu,
                color: "rgba(255,255,255,0.08)",
              },
            ].map((seg, i) => (
              <div
                key={i}
                style={{
                  height: 20,
                  flex: Math.max(seg.gb, 0.5),
                  background: seg.color,
                  borderRadius: i === 0 ? "4px 0 0 4px" : i === 4 ? "0 4px 4px 0" : 0,
                  opacity: 0.6,
                }}
                title={`${seg.label}: ${seg.gb.toFixed(1)}GB`}
              />
            ))}
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "8px 16px",
              fontFamily: mono,
              fontSize: 10,
              color: "rgba(255,255,255,0.4)",
            }}
          >
            {[
              { label: "Modello", gb: modelSize, color: "#6B8DD6" },
              {
                label: "LoRA adapters",
                gb: trainableParams * 0.004,
                color: "#76b947",
              },
              {
                label: "Optimizer states",
                gb: trainableParams * 0.016,
                color: "#D4943A",
              },
              {
                label: "Activations + grad",
                gb: useQuantization ? 3 : 6,
                color: "#E8654A",
              },
            ].map((s, i) => (
              <span key={i}>
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: s.color,
                    marginRight: 4,
                    opacity: 0.6,
                  }}
                />
                {s.label}: {s.gb.toFixed(1)}GB
              </span>
            ))}
            <span style={{ color: "rgba(255,255,255,0.25)" }}>
              Usato: ~{vramPerGpu}GB / 48GB
            </span>
          </div>
        </div>

        {/* Comparison */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 10,
          }}
        >
          {[
            {
              label: "Nostro test (CPU)",
              time: "4 min",
              samples: "1.6K",
              quality: "PoC",
            },
            {
              label: `2× RTX 6000 Ada`,
              time: `${totalHours < 1 ? Math.round(totalMin) + " min" : totalHours.toFixed(1) + "h"}`,
              samples: "130K",
              quality: "Robusto",
            },
            {
              label: "Cluster 8× A100",
              time: "~20 min",
              samples: "130K",
              quality: "Produzione",
            },
          ].map((item, i) => (
            <div
              key={i}
              style={{
                background:
                  i === 1
                    ? "rgba(118,185,71,0.05)"
                    : "rgba(255,255,255,0.02)",
                borderRadius: 10,
                padding: 16,
                border: `1px solid ${i === 1 ? "rgba(118,185,71,0.15)" : "rgba(255,255,255,0.04)"}`,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontFamily: mono,
                  fontSize: 9,
                  color:
                    i === 1 ? "#76b947" : "rgba(255,255,255,0.3)",
                  letterSpacing: "0.08em",
                  marginBottom: 8,
                }}
              >
                {item.label}
              </div>
              <div style={{ fontSize: 22, marginBottom: 4 }}>{item.time}</div>
              <div
                style={{
                  fontFamily: mono,
                  fontSize: 10,
                  color: "rgba(255,255,255,0.3)",
                }}
              >
                {item.samples} samples • {item.quality}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
