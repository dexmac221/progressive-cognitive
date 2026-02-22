import { useState } from "react";

const PHASES = [
  {
    id: 0,
    title: "Fase 1 — Fondamenta",
    subtitle: "Apprendimento grezzo",
    metaphor: "Il bambino che impara l'aritmetica",
    description:
      "Il modello apprende conoscenza esatta e granulare: fatti, calcoli, pattern di base. Tutto è esplicito nei pesi. Alta energia, bassa astrazione.",
    color: "#E8654A",
    bgGrad: "linear-gradient(135deg, #1a0a08 0%, #2d1210 100%)",
    knowledge: 95,
    intuition: 5,
    toolUse: 0,
    details: [
      "Training su dati grezzi e compiti elementari",
      "I pesi codificano conoscenza esplicita",
      "Nessun accesso a tool esterni",
      "Alta ridondanza, molti parametri per compiti semplici",
    ],
  },
  {
    id: 1,
    title: "Fase 2 — Consolidamento",
    subtitle: "Compressione in intuizione",
    metaphor: "Lo studente che passa all'algebra",
    description:
      "La conoscenza esatta viene compressa. I circuiti si astraggono: il calcolo diventa senso numerico. Emerge la capacità di stimare, approssimare, 'sentire' se qualcosa torna.",
    color: "#D4943A",
    bgGrad: "linear-gradient(135deg, #14100a 0%, #2a2010 100%)",
    knowledge: 55,
    intuition: 40,
    toolUse: 5,
    details: [
      "Pruning guidato: rimuovi circuiti di calcolo esatto",
      "Penalizza risposte precise, premia stime plausibili",
      "I pesi si riorganizzano in attrattori compressi",
      "Simile al sonno: consolidazione e dimenticanza strutturata",
    ],
  },
  {
    id: 2,
    title: "Fase 3 — Delega",
    subtitle: "Integrazione dei tool",
    metaphor: "L'adulto che usa la calcolatrice",
    description:
      "Il modello impara quando e come delegare ai tool deterministici. Il calcolo interno è sostituito da orchestrazione esterna. I parametri liberati vengono riallocati al ragionamento di alto livello.",
    color: "#5BA88C",
    bgGrad: "linear-gradient(135deg, #080f0d 0%, #0f2a22 100%)",
    knowledge: 25,
    intuition: 45,
    toolUse: 30,
    details: [
      "Introduzione di tool: calcolatrice, DB, search",
      "Training: usa il tool invece di calcolare internamente",
      "Il modello impara a scegliere il tool giusto",
      "Validazione intuitiva dei risultati del tool",
    ],
  },
  {
    id: 3,
    title: "Fase 4 — Orchestrazione",
    subtitle: "Intelligenza emergente",
    metaphor: "L'esperto che vede il bug senza leggere il codice",
    description:
      "Il modello è un orchestratore: intuizione forte per capire cosa serve, tool per eseguire, senso critico per validare. Minimo spreco di parametri, massima efficacia. Sistema complesso adattivo.",
    color: "#6B8DD6",
    bgGrad: "linear-gradient(135deg, #080c14 0%, #101a2d 100%)",
    knowledge: 10,
    intuition: 50,
    toolUse: 40,
    details: [
      "Nucleo intuitivo leggero + tool deterministici",
      "Meno parametri → meno allucinazioni fattuali",
      "Meno calcolo → meno consumo energetico",
      "Emergenza di 'senso' non misurabile ma funzionale",
    ],
  },
];

function Bar({ label, value, color, delay }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          fontFamily: "'JetBrains Mono', monospace",
          color: "rgba(255,255,255,0.5)",
          marginBottom: 3,
        }}
      >
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div
        style={{
          height: 6,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${value}%`,
            background: color,
            borderRadius: 3,
            transition: "width 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
            transitionDelay: `${delay}ms`,
          }}
        />
      </div>
    </div>
  );
}

function ChaosDiagram({ phase }) {
  const nodes = [];
  const connections = [];
  const cx = 140;
  const cy = 90;

  const knowledgeNodes = Math.round((phase.knowledge / 100) * 12);
  const intuitionNodes = Math.round((phase.intuition / 100) * 8);
  const toolNodes = Math.round((phase.toolUse / 100) * 6);

  for (let i = 0; i < knowledgeNodes; i++) {
    const angle = (i / Math.max(knowledgeNodes, 1)) * Math.PI * 2;
    const r = 30 + Math.sin(i * 1.7) * 15;
    nodes.push({
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
      type: "knowledge",
      r: 3,
    });
  }

  for (let i = 0; i < intuitionNodes; i++) {
    const angle = (i / Math.max(intuitionNodes, 1)) * Math.PI * 2 + 0.3;
    const r = 50 + Math.cos(i * 2.3) * 10;
    nodes.push({
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
      type: "intuition",
      r: 5 + phase.intuition * 0.04,
    });
  }

  for (let i = 0; i < toolNodes; i++) {
    const angle = (i / Math.max(toolNodes, 1)) * Math.PI * 2 + 1.1;
    const r = 65;
    nodes.push({
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
      type: "tool",
      r: 6,
    });
  }

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 45) {
        connections.push({ from: nodes[i], to: nodes[j], dist });
      }
    }
  }

  const typeColors = {
    knowledge: phase.color,
    intuition: "#fff",
    tool: "#88aaff",
  };

  return (
    <svg
      width="280"
      height="180"
      viewBox="0 0 280 180"
      style={{ display: "block" }}
    >
      {connections.map((c, i) => (
        <line
          key={`c-${i}`}
          x1={c.from.x}
          y1={c.from.y}
          x2={c.to.x}
          y2={c.to.y}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={0.5}
        />
      ))}
      {nodes.map((n, i) => (
        <g key={`n-${i}`}>
          <circle
            cx={n.x}
            cy={n.y}
            r={n.r}
            fill={
              n.type === "tool"
                ? "none"
                : typeColors[n.type]
            }
            stroke={typeColors[n.type]}
            strokeWidth={n.type === "tool" ? 1.5 : 0}
            opacity={n.type === "knowledge" ? 0.5 : 0.8}
          />
          {n.type === "intuition" && (
            <circle
              cx={n.x}
              cy={n.y}
              r={n.r + 4}
              fill="none"
              stroke="rgba(255,255,255,0.1)"
              strokeWidth={0.5}
            />
          )}
        </g>
      ))}
      <text
        x={20}
        y={175}
        fill="rgba(255,255,255,0.25)"
        fontSize={8}
        fontFamily="'JetBrains Mono', monospace"
      >
        ● conoscenza {"  "}○ intuizione {"  "}◇ tool
      </text>
    </svg>
  );
}

function FlowArrow({ color }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "12px 0",
      }}
    >
      <svg width="40" height="40" viewBox="0 0 40 40">
        <path
          d="M20 5 L20 28 M12 22 L20 30 L28 22"
          stroke={color}
          strokeWidth="1.5"
          fill="none"
          opacity="0.4"
        />
        <text
          x="20"
          y="38"
          textAnchor="middle"
          fill={color}
          fontSize="7"
          fontFamily="'JetBrains Mono', monospace"
          opacity="0.5"
        >
          comprimi
        </text>
      </svg>
    </div>
  );
}

export default function CognitiveArchitecture() {
  const [activePhase, setActivePhase] = useState(0);
  const phase = PHASES[activePhase];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0c",
        color: "#e8e8e8",
        fontFamily:
          "'Instrument Serif', 'Georgia', serif",
        padding: "40px 20px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400&display=swap"
        rel="stylesheet"
      />

      <div style={{ maxWidth: 720, width: "100%" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 48 }}>
          <h1
            style={{
              fontSize: 32,
              fontWeight: 400,
              margin: 0,
              letterSpacing: "-0.02em",
              lineHeight: 1.2,
            }}
          >
            Architettura Cognitiva
            <br />
            <span style={{ fontStyle: "italic", opacity: 0.5, fontSize: 26 }}>
              Progressiva
            </span>
          </h1>
          <p
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              color: "rgba(255,255,255,0.3)",
              marginTop: 16,
              letterSpacing: "0.05em",
            }}
          >
            DA CONOSCENZA ESPLICITA A INTUIZIONE COMPRESSA
          </p>
        </div>

        {/* Phase selector */}
        <div
          style={{
            display: "flex",
            gap: 2,
            marginBottom: 32,
            borderRadius: 8,
            overflow: "hidden",
          }}
        >
          {PHASES.map((p, i) => (
            <button
              key={p.id}
              onClick={() => setActivePhase(i)}
              style={{
                flex: 1,
                padding: "14px 8px",
                background:
                  i === activePhase
                    ? p.color + "22"
                    : "rgba(255,255,255,0.03)",
                border: "none",
                borderBottom: `2px solid ${i === activePhase ? p.color : "transparent"}`,
                color:
                  i === activePhase ? p.color : "rgba(255,255,255,0.3)",
                cursor: "pointer",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                letterSpacing: "0.03em",
                transition: "all 0.3s ease",
              }}
            >
              <div style={{ fontSize: 18, marginBottom: 4 }}>
                {i === 0 ? "△" : i === 1 ? "◐" : i === 2 ? "◈" : "✦"}
              </div>
              Fase {i + 1}
            </button>
          ))}
        </div>

        {/* Main content */}
        <div
          style={{
            background: phase.bgGrad,
            borderRadius: 16,
            border: `1px solid ${phase.color}15`,
            padding: 32,
            transition: "all 0.5s ease",
          }}
        >
          {/* Title section */}
          <div style={{ marginBottom: 28 }}>
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                color: phase.color,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: 8,
              }}
            >
              {phase.subtitle}
            </div>
            <h2
              style={{
                fontSize: 26,
                fontWeight: 400,
                margin: 0,
                marginBottom: 8,
              }}
            >
              {phase.title}
            </h2>
            <p
              style={{
                fontStyle: "italic",
                color: "rgba(255,255,255,0.4)",
                fontSize: 15,
                margin: 0,
              }}
            >
              « {phase.metaphor} »
            </p>
          </div>

          {/* Two columns */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 28,
            }}
          >
            {/* Left: Chaos diagram */}
            <div>
              <div
                style={{
                  background: "rgba(0,0,0,0.3)",
                  borderRadius: 12,
                  padding: 12,
                  border: "1px solid rgba(255,255,255,0.04)",
                }}
              >
                <ChaosDiagram phase={phase} />
              </div>
              <div style={{ marginTop: 16 }}>
                <Bar
                  label="Conoscenza esplicita"
                  value={phase.knowledge}
                  color={phase.color}
                  delay={0}
                />
                <Bar
                  label="Intuizione compressa"
                  value={phase.intuition}
                  color="#ffffff"
                  delay={100}
                />
                <Bar
                  label="Delega a tool"
                  value={phase.toolUse}
                  color="#88aaff"
                  delay={200}
                />
              </div>
            </div>

            {/* Right: Description + details */}
            <div>
              <p
                style={{
                  fontSize: 14,
                  lineHeight: 1.7,
                  color: "rgba(255,255,255,0.7)",
                  margin: 0,
                  marginBottom: 20,
                }}
              >
                {phase.description}
              </p>
              <div
                style={{
                  borderTop: "1px solid rgba(255,255,255,0.06)",
                  paddingTop: 16,
                }}
              >
                {phase.details.map((d, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      gap: 10,
                      marginBottom: 10,
                      fontSize: 12,
                      fontFamily: "'JetBrains Mono', monospace",
                      color: "rgba(255,255,255,0.45)",
                      lineHeight: 1.5,
                    }}
                  >
                    <span style={{ color: phase.color, flexShrink: 0 }}>→</span>
                    <span>{d}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Flow arrows between phases */}
        {activePhase < 3 && (
          <div style={{ display: "flex", justifyContent: "center" }}>
            <FlowArrow color={phase.color} />
          </div>
        )}

        {/* Principle box */}
        <div
          style={{
            marginTop: activePhase < 3 ? 0 : 24,
            background: "rgba(255,255,255,0.02)",
            borderRadius: 12,
            padding: 24,
            border: "1px solid rgba(255,255,255,0.05)",
            textAlign: "center",
          }}
        >
          <div
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9,
              color: "rgba(255,255,255,0.25)",
              letterSpacing: "0.15em",
              marginBottom: 12,
            }}
          >
            PRINCIPIO GUIDA
          </div>
          <p
            style={{
              fontSize: 18,
              fontStyle: "italic",
              color: "rgba(255,255,255,0.6)",
              margin: 0,
              lineHeight: 1.6,
            }}
          >
            La conoscenza non scompare — collassa in attrattori.
            <br />
            L'intuizione è il residuo compresso dell'esperienza.
            <br />
            <span style={{ fontSize: 14, opacity: 0.5 }}>
              Come nei sistemi caotici, l'ordine emerge dalla complessità.
            </span>
          </p>
        </div>

        {/* Architecture summary */}
        <div
          style={{
            marginTop: 24,
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 12,
          }}
        >
          {[
            {
              icon: "◯",
              label: "Nucleo intuitivo",
              desc: "Pochi parametri, alta astrazione. Sa cosa serve e cosa aspettarsi.",
            },
            {
              icon: "⬡",
              label: "Tool deterministici",
              desc: "Calcolo, ricerca, database. Esatti, veloci, affidabili.",
            },
            {
              icon: "◇",
              label: "Validazione",
              desc: "Il senso critico: 'questo risultato ha senso?' — l'intuito dell'esperto.",
            },
          ].map((item, i) => (
            <div
              key={i}
              style={{
                background: "rgba(255,255,255,0.02)",
                borderRadius: 10,
                padding: 18,
                border: "1px solid rgba(255,255,255,0.04)",
              }}
            >
              <div
                style={{ fontSize: 22, marginBottom: 8, opacity: 0.6 }}
              >
                {item.icon}
              </div>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  color: "rgba(255,255,255,0.5)",
                  marginBottom: 6,
                  letterSpacing: "0.05em",
                }}
              >
                {item.label}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "rgba(255,255,255,0.35)",
                  lineHeight: 1.5,
                }}
              >
                {item.desc}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
