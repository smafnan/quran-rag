import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2 } from "lucide-react";
import { surahName } from "./surahs";

type Result = {
  ref: string;
  text: string;
  arabic: string;
  score: number;
  matched: string[];
  has_tafseer: boolean;
};
type Correction = { from: string; to: string };
type Expansion = { term: string; also_searched: string[] };
type Interpretation = {
  original: string;
  effective: string;
  corrections: Correction[];
  expanded: Expansion[];
  suggestions: string[];
  needs_confirmation: boolean;
};
// named SearchResponse so it does not shadow the DOM's Response type
type SearchResponse = {
  found: boolean;
  source: string;
  count: number;
  results: Result[];
  needs_confirmation?: boolean;
  suggestions?: string[];
  interpretation?: Interpretation;
};
type Explanation = { loading: boolean; text?: string; error?: string };
type TafseerPanel = { loading: boolean; text?: string; error?: string };
type Theme = "light" | "dark";

const SAMPLES = ["patience through hardship", "the mercy of Allah", "gratitude", "justice"];
const PAGE = 25;
const REPO = "https://github.com/smafnan/quran-rag";

const MONO = "'IBM Plex Mono', monospace";
const SERIF = "'Newsreader', Georgia, serif";
const AR_SAMPLE = "ٱقْرَأْ";
const ARABIC_FONTS = [
  { label: "Amiri", css: "'Amiri', serif" },
  { label: "Scheherazade", css: "'Scheherazade New', serif" },
  { label: "Noto Naskh", css: "'Noto Naskh Arabic', serif" },
  { label: "Markazi", css: "'Markazi Text', serif" },
  { label: "Aref Ruqaa", css: "'Aref Ruqaa', serif" },
];

// Empty when the API is served from the same origin (local dev, or the single
// container on Render). Set VITE_API_URL at build time to point a statically
// hosted frontend at a separately hosted backend.
const API = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

// A free-tier backend sleeps when idle and takes ~50s to wake, so give requests
// room and tell the user what is happening instead of failing at the default.
const WAKE_TIMEOUT_MS = 90_000;

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), WAKE_TIMEOUT_MS);
  try {
    return await fetch(`${API}${path}`, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

function refKey(ref: string): [number, number] {
  const [c, v] = ref.split(":").map(Number);
  return [c, v];
}

function matchLabel(matched: string[]): string {
  if (matched.includes("keyword")) return "word match";
  if (matched.includes("semantic")) return "related by meaning";
  return "related";
}

// ---- shared style fragments (colours come from the CSS variables in index.css)
const monoEyebrow: React.CSSProperties = {
  fontFamily: MONO,
  fontSize: 10,
  letterSpacing: ".14em",
  textTransform: "uppercase",
  color: "var(--ink-faint)",
};
const pillBtn: React.CSSProperties = {
  fontFamily: MONO,
  fontSize: 11,
  letterSpacing: ".1em",
  textTransform: "uppercase",
  color: "var(--brass)",
  background: "transparent",
  border: "1px solid var(--line-strong)",
  borderRadius: 999,
  padding: "9px 16px",
  whiteSpace: "nowrap",
  cursor: "pointer",
  display: "inline-flex",
  alignItems: "center",
  gap: 7,
};

function Diamonds({ n }: { n: number }) {
  const filled = Math.max(0, Math.min(4, n));
  return (
    <span style={{ letterSpacing: ".15em", color: "var(--brass)", fontSize: 12 }}>
      {"◆".repeat(filled) + "◇".repeat(4 - filled)}
    </span>
  );
}

export default function App() {
  const [q, setQ] = useState("");
  const [asked, setAsked] = useState("");
  const [resp, setResp] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState<number | null>(null);
  const [explainAvailable, setExplainAvailable] = useState(true);
  const [explanations, setExplanations] = useState<Record<string, Explanation>>({});
  const [tafseers, setTafseers] = useState<Record<string, TafseerPanel>>({});
  const [sortBy, setSortBy] = useState<"relevance" | "quran">("relevance");
  const [visible, setVisible] = useState(PAGE);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [waking, setWaking] = useState(false);
  const [theme, setTheme] = useState<Theme>("light");
  const [arFont, setArFont] = useState(ARABIC_FONTS[0].css);
  const [arScale, setArScale] = useState(1);
  const reqSeq = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    apiFetch("/api/info")
      .then((r) => r.json())
      .then((d) => {
        setCount(d.passages);
        setExplainAvailable(!!d.explain_available);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // ⌘K / Ctrl+K focuses the search box, as the design's hint promises.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /** Search the literal text, overriding any spelling correction. */
  async function askExact(question: string) {
    return ask(question, true);
  }

  async function ask(question?: string, exact = false) {
    const text = (question ?? q).trim();
    if (!text || loading) return;
    const seq = ++reqSeq.current;
    setQ(text);
    setLoading(true);
    setExplanations({});
    setTafseers({});
    setVisible(PAGE);
    // a sleeping free-tier backend takes ~50s to wake; say so rather than
    // leaving a spinner that looks like the app has hung
    const wakeHint = setTimeout(() => {
      if (seq === reqSeq.current) setWaking(true);
    }, 4000);
    try {
      const r = await apiFetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, mode: "all", exact }),
      });
      const data = await r.json();
      if (seq !== reqSeq.current) return; // a newer search superseded this one
      if (!r.ok) {
        // a rate limit or upstream error must not read as "no verses found"
        setSearchError(data?.detail || `Search failed (${r.status})`);
        setResp(null);
        setAsked(text);
        return;
      }
      setSearchError(null);
      setResp(data);
      setAsked(text);
    } catch (e) {
      if (seq === reqSeq.current) {
        const aborted = (e as { name?: string })?.name === "AbortError";
        setSearchError(
          aborted
            ? "The server took too long to wake up. It sleeps when idle — try once more."
            : "Could not reach the search service."
        );
        setResp(null);
        setAsked(text);
      }
    } finally {
      clearTimeout(wakeHint);
      if (seq === reqSeq.current) {
        setLoading(false);
        setWaking(false);
      }
    }
  }

  async function toggleTafseer(ref: string) {
    if (tafseers[ref]) {
      if (!tafseers[ref].loading) {
        setTafseers((prev) => {
          const { [ref]: _drop, ...rest } = prev;
          return rest;
        });
      }
      return;
    }
    setTafseers((prev) => ({ ...prev, [ref]: { loading: true } }));
    try {
      const r = await apiFetch(`/api/tafseer?ref=${encodeURIComponent(ref)}`);
      if (!r.ok) throw new Error(`${r.status}`);
      const d = await r.json();
      setTafseers((prev) => ({
        ...prev,
        [ref]: {
          loading: false,
          text: d.tafseer || "No tafsīr is available for this verse in the loaded corpus.",
        },
      }));
    } catch {
      setTafseers((prev) => ({
        ...prev,
        [ref]: { loading: false, error: "Could not load the tafsīr." },
      }));
    }
  }

  async function explain(ref: string) {
    if (explanations[ref]) {
      if (!explanations[ref].loading) {
        setExplanations((prev) => {
          const { [ref]: _drop, ...rest } = prev;
          return rest;
        });
      }
      return;
    }
    setExplanations((prev) => ({ ...prev, [ref]: { loading: true } }));
    try {
      const r = await apiFetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref, question: asked }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({} as { detail?: string }));
        setExplanations((prev) => ({
          ...prev,
          [ref]: { loading: false, error: d.detail || `Request failed (${r.status})` },
        }));
        return;
      }
      const d = await r.json();
      setExplanations((prev) => ({ ...prev, [ref]: { loading: false, text: d.explanation } }));
    } catch {
      setExplanations((prev) => ({
        ...prev,
        [ref]: { loading: false, error: "Could not reach the explanation service." },
      }));
    }
  }

  const ordered = useMemo(() => {
    if (!resp?.results) return [];
    if (sortBy === "relevance") return resp.results;
    return [...resp.results].sort((a, b) => {
      const [ca, va] = refKey(a.ref);
      const [cb, vb] = refKey(b.ref);
      return ca - cb || va - vb;
    });
  }, [resp, sortBy]);

  const maxScore = useMemo(
    () => (resp?.results?.length ? Math.max(...resp.results.map((r) => r.score)) : 0),
    [resp]
  );
  const shown = ordered.slice(0, visible);
  const interp = resp?.interpretation;
  const hasCorrection = !!interp && !resp?.needs_confirmation && interp.corrections.length > 0;
  const hasExpansion =
    !!interp && !resp?.needs_confirmation && interp.corrections.length === 0 && interp.expanded.length > 0;
  const resultsRegionVisible =
    resp !== null || searchError !== null || (waking && loading);

  const segStyle = (active: boolean): React.CSSProperties => ({
    fontFamily: MONO,
    fontSize: 10,
    letterSpacing: ".06em",
    textTransform: "uppercase",
    border: "none",
    borderRadius: 999,
    padding: "6px 11px",
    cursor: "pointer",
    transition: "all .2s",
    color: active ? "var(--bg)" : "var(--ink-soft)",
    background: active ? "var(--brass)" : "transparent",
  });

  const rootStyle = {
    minHeight: "100vh",
    background: "var(--bg2)",
    color: "var(--ink)",
    fontFamily: SERIF,
    padding: "clamp(20px,4vw,54px) clamp(16px,4vw,54px)",
    ["--ar-font" as string]: arFont,
    ["--ar-scale" as string]: arScale,
  } as React.CSSProperties;

  return (
    <div style={rootStyle}>
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <div
          style={{
            background: "var(--bg)",
            border: "1px solid var(--line)",
            borderRadius: 26,
            overflow: "hidden",
            boxShadow: "0 50px 100px -55px rgba(0,0,0,.5)",
          }}
        >
          {/* ── Top bar ─────────────────────────────────────────────── */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 16,
              padding: "17px clamp(18px,3vw,30px)",
              borderBottom: "1px solid var(--line)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div
                style={{
                  width: 26,
                  height: 26,
                  borderRadius: 7,
                  background: "var(--brass)",
                  transform: "rotate(45deg)",
                  flex: "none",
                }}
              />
              <div style={{ lineHeight: 1.05 }}>
                <div style={{ fontFamily: SERIF, fontWeight: 600, fontSize: 18, color: "var(--ink)" }}>
                  Qur'ān <span style={{ color: "var(--brass)" }}>RAG</span>
                </div>
                <div
                  style={{
                    fontFamily: MONO,
                    fontSize: 9,
                    letterSpacing: ".16em",
                    textTransform: "uppercase",
                    color: "var(--ink-faint)",
                  }}
                >
                  grounded search
                </div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
              <a
                className="dc-navlink"
                href={REPO}
                target="_blank"
                rel="noreferrer"
                style={{ fontFamily: SERIF, fontSize: 14, color: "var(--ink-soft)" }}
              >
                GitHub
              </a>
              <div
                style={{
                  display: "inline-flex",
                  padding: 3,
                  gap: 3,
                  border: "1px solid var(--line-strong)",
                  borderRadius: 999,
                  background: "var(--surface2)",
                }}
              >
                <button onClick={() => setTheme("light")} style={segStyle(theme === "light")}>
                  Light
                </button>
                <button onClick={() => setTheme("dark")} style={segStyle(theme === "dark")}>
                  Dark
                </button>
              </div>
            </div>
          </div>

          {/* ── Hero ────────────────────────────────────────────────── */}
          <div
            style={{
              position: "relative",
              overflow: "hidden",
              padding: "clamp(48px,7vw,78px) clamp(18px,3vw,30px) clamp(44px,6vw,60px)",
              textAlign: "center",
            }}
          >
            <div
              aria-hidden="true"
              style={{
                position: "absolute",
                top: "-34%",
                left: "50%",
                width: 720,
                height: 560,
                background: "radial-gradient(50% 50% at 50% 50%,var(--glow),transparent 70%)",
                filter: "blur(26px)",
                pointerEvents: "none",
                animation: "auroraDrift 18s ease-in-out infinite",
              }}
            />
            <div style={{ position: "relative" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 14,
                  marginBottom: 22,
                }}
              >
                <span
                  style={{
                    width: 46,
                    height: 1,
                    background: "linear-gradient(90deg,transparent,var(--brass))",
                    opacity: 0.55,
                  }}
                />
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: 11,
                    letterSpacing: ".24em",
                    textTransform: "uppercase",
                    color: "var(--brass)",
                  }}
                >
                  Grounded Qur'ānic search
                </span>
                <span
                  style={{
                    width: 46,
                    height: 1,
                    background: "linear-gradient(90deg,var(--brass),transparent)",
                    opacity: 0.55,
                  }}
                />
              </div>
              <h1
                style={{
                  fontFamily: SERIF,
                  fontWeight: 500,
                  fontSize: "clamp(34px,4.4vw,58px)",
                  lineHeight: 1.05,
                  letterSpacing: "-.015em",
                  color: "var(--ink)",
                  margin: "0 auto 20px",
                  maxWidth: "15ch",
                }}
              >
                Ask a question.
                <br />
                Receive <span style={{ fontStyle: "italic" }}>the verses.</span>
              </h1>
              <p
                style={{
                  fontFamily: SERIF,
                  fontSize: 19,
                  lineHeight: 1.6,
                  color: "var(--ink-soft)",
                  maxWidth: "50ch",
                  margin: "0 auto 14px",
                }}
              >
                Search the Qur'ān by topic and read every related verse — in Arabic and English, each
                one cited.
              </p>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 9,
                  maxWidth: "52ch",
                  margin: "0 auto 32px",
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    background: "var(--brass)",
                    transform: "rotate(45deg)",
                    flex: "none",
                  }}
                />
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: 12,
                    lineHeight: 1.5,
                    letterSpacing: ".01em",
                    color: "var(--ink-faint)",
                  }}
                >
                  An AI study aid — it can make mistakes, so double-check every answer against the
                  source and trusted scholarship.
                </span>
              </div>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  ask();
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  maxWidth: 600,
                  margin: "0 auto",
                  background: "var(--surface)",
                  border: "1px solid var(--line-strong)",
                  borderRadius: 999,
                  padding: "8px 8px 8px 22px",
                  boxShadow: "0 22px 60px -30px var(--glow)",
                }}
              >
                <span style={{ color: "var(--brass)", fontSize: 18, lineHeight: 1 }}>⌕</span>
                <input
                  ref={inputRef}
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="Enter a topic — patience, Musa, mercy…"
                  aria-label="Search the Qur'ān by topic"
                  style={{
                    flex: 1,
                    minWidth: 0,
                    border: "none",
                    background: "transparent",
                    outline: "none",
                    fontFamily: SERIF,
                    fontSize: 17,
                    color: "var(--ink)",
                  }}
                />
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: 10,
                    letterSpacing: ".06em",
                    color: "var(--ink-faint)",
                    border: "1px solid var(--line)",
                    borderRadius: 7,
                    padding: "5px 8px",
                  }}
                >
                  ⌘K
                </span>
                <button
                  type="submit"
                  disabled={loading}
                  className="dc-solid"
                  style={{
                    fontFamily: MONO,
                    fontSize: 12,
                    letterSpacing: ".1em",
                    textTransform: "uppercase",
                    color: "var(--bg)",
                    background: "var(--brass)",
                    border: "none",
                    borderRadius: 999,
                    padding: "12px 22px",
                    cursor: "pointer",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  {loading && <Loader2 className="animate-spin" size={14} />}
                  Search
                </button>
              </form>

              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 9,
                  justifyContent: "center",
                  marginTop: 20,
                }}
              >
                {SAMPLES.map((c) => (
                  <button
                    key={c}
                    onClick={() => ask(c)}
                    className="dc-chip"
                    style={{
                      fontFamily: SERIF,
                      fontSize: 14,
                      color: "var(--ink-soft)",
                      border: "1px solid var(--line)",
                      borderRadius: 999,
                      padding: "7px 15px",
                      background: "var(--surface)",
                      cursor: "pointer",
                    }}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Results region ──────────────────────────────────────── */}
          {resultsRegionVisible && (
            <div style={{ maxWidth: 820, margin: "0 auto", padding: "6px clamp(18px,3vw,30px) 44px" }}>
              {/* cold-start notice */}
              {waking && loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 10,
                    marginTop: 22,
                    padding: "13px 16px",
                    background: "var(--brass-soft)",
                    border: "1px solid var(--line)",
                    borderRadius: 14,
                    fontFamily: MONO,
                    fontSize: 12,
                    color: "var(--ink-soft)",
                  }}
                >
                  <Loader2 className="animate-spin" size={14} style={{ color: "var(--brass)" }} />
                  Waking the server — it sleeps when idle, so this first search can take up to a minute.
                </motion.div>
              )}

              {/* search error (rate limit, upstream failure) */}
              {searchError && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    marginTop: 22,
                    padding: "20px 22px",
                    background: "var(--surface)",
                    border: "1px solid var(--line-strong)",
                    borderLeft: "3px solid var(--danger)",
                    borderRadius: 16,
                    textAlign: "center",
                  }}
                >
                  <p style={{ fontFamily: SERIF, fontSize: 17, color: "var(--ink)", margin: 0 }}>
                    {searchError}
                  </p>
                  <p style={{ ...monoEyebrow, marginTop: 8, textTransform: "none", letterSpacing: ".02em" }}>
                    This is a limit or a connection problem — not a statement about the text.
                  </p>
                </motion.div>
              )}

              {/* spelling understood: correction or expansion */}
              {(hasCorrection || hasExpansion) && (
                <div
                  style={{
                    marginTop: 22,
                    padding: "13px 16px",
                    background: "var(--surface2)",
                    border: "1px solid var(--line)",
                    borderLeft: "3px solid var(--brass)",
                    borderRadius: 14,
                    fontFamily: SERIF,
                    fontSize: 15,
                    color: "var(--ink-soft)",
                  }}
                >
                  {hasCorrection ? (
                    <span>
                      Showing results for{" "}
                      <strong style={{ color: "var(--ink)", fontStyle: "italic" }}>
                        {interp!.effective}
                      </strong>
                      .{" "}
                      <button
                        onClick={() => askExact(interp!.original)}
                        className="dc-underline"
                        style={{
                          background: "transparent",
                          border: "none",
                          padding: 0,
                          cursor: "pointer",
                          fontFamily: SERIF,
                          fontSize: 15,
                          color: "var(--brass)",
                        }}
                      >
                        Search instead for "{interp!.original}"
                      </button>
                    </span>
                  ) : (
                    <span>
                      Also searched{" "}
                      {interp!.expanded
                        .flatMap((e) => e.also_searched)
                        .map((t, i, arr) => (
                          <span key={t}>
                            <strong style={{ color: "var(--ink)", fontStyle: "italic" }}>{t}</strong>
                            {i < arr.length - 1 ? ", " : ""}
                          </span>
                        ))}{" "}
                      — the same name is spelled differently across translations.
                    </span>
                  )}
                </div>
              )}

              {/* genuinely ambiguous: ask rather than guess */}
              {resp?.needs_confirmation && (resp.suggestions?.length ?? 0) > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    marginTop: 30,
                    background: "var(--surface)",
                    border: "1px dashed var(--line-strong)",
                    borderRadius: 20,
                    padding: 30,
                    textAlign: "center",
                  }}
                >
                  <p style={{ fontFamily: SERIF, fontSize: 18, color: "var(--ink)", margin: "0 0 16px" }}>
                    No verses matched "{asked}". Did you mean:
                  </p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 9, justifyContent: "center" }}>
                    {resp.suggestions!.map((s) => (
                      <button
                        key={s}
                        onClick={() => ask(s)}
                        className="dc-chip"
                        style={{
                          fontFamily: SERIF,
                          fontSize: 15,
                          color: "var(--ink)",
                          border: "1px solid var(--line-strong)",
                          borderRadius: 999,
                          padding: "8px 16px",
                          background: "var(--surface2)",
                          cursor: "pointer",
                        }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                  <p style={{ ...monoEyebrow, marginTop: 16, textTransform: "none", letterSpacing: ".02em" }}>
                    Picking one searches it; nothing has been assumed on your behalf.
                  </p>
                </motion.div>
              )}

              {/* results list */}
              {resp?.found && (
                <>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "baseline",
                      justifyContent: "space-between",
                      gap: 16,
                      borderTop: "1px solid var(--line)",
                      paddingTop: 28,
                      marginTop: 22,
                      flexWrap: "wrap",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
                      <span style={{ ...monoEyebrow, fontSize: 11 }}>
                        {resp.count} connected verse{resp.count === 1 ? "" : "s"}
                      </span>
                      <span
                        style={{
                          fontFamily: MONO,
                          fontSize: 12,
                          color: "var(--brass)",
                          border: "1px solid var(--line-strong)",
                          borderRadius: 999,
                          padding: "3px 12px",
                        }}
                      >
                        {asked}
                      </span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ ...monoEyebrow, fontSize: 11 }}>Sort</span>
                      {(["relevance", "quran"] as const).map((s) => (
                        <button
                          key={s}
                          onClick={() => setSortBy(s)}
                          style={{
                            fontFamily: MONO,
                            fontSize: 10,
                            letterSpacing: ".06em",
                            textTransform: "uppercase",
                            border: "none",
                            borderRadius: 999,
                            padding: "5px 10px",
                            cursor: "pointer",
                            background: sortBy === s ? "var(--brass-soft)" : "transparent",
                            color: sortBy === s ? "var(--brass)" : "var(--ink-faint)",
                          }}
                        >
                          {s === "relevance" ? "Relevance" : "Qur'ān order"}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Arabic font + size controls */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 14,
                      flexWrap: "wrap",
                      marginTop: 18,
                      padding: "11px 14px",
                      background: "var(--surface2)",
                      border: "1px solid var(--line)",
                      borderRadius: 14,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                      <span style={{ ...monoEyebrow, whiteSpace: "nowrap" }}>Arabic font</span>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {ARABIC_FONTS.map((f) => {
                          const active = arFont === f.css;
                          return (
                            <button
                              key={f.label}
                              onClick={() => setArFont(f.css)}
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 8,
                                border: `1px solid ${active ? "var(--brass)" : "var(--line-strong)"}`,
                                borderRadius: 10,
                                padding: "5px 11px",
                                background: active ? "var(--brass-soft)" : "transparent",
                                cursor: "pointer",
                                transition: "all .2s",
                              }}
                            >
                              <span
                                dir="rtl"
                                lang="ar"
                                style={{ fontFamily: f.css, fontSize: 20, lineHeight: 1.7, color: "var(--arabic)" }}
                              >
                                {AR_SAMPLE}
                              </span>
                              <span
                                style={{
                                  fontFamily: MONO,
                                  fontSize: 9,
                                  letterSpacing: ".05em",
                                  textTransform: "uppercase",
                                  color: "var(--ink-soft)",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {f.label}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <span style={monoEyebrow}>Size</span>
                      <div
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          border: "1px solid var(--line-strong)",
                          borderRadius: 10,
                          overflow: "hidden",
                        }}
                      >
                        <button
                          onClick={() => setArScale((s) => Math.max(0.8, +(s - 0.1).toFixed(2)))}
                          style={{
                            border: "none",
                            background: "transparent",
                            cursor: "pointer",
                            padding: "6px 13px",
                            fontFamily: SERIF,
                            fontSize: 14,
                            color: "var(--brass)",
                            lineHeight: 1,
                          }}
                        >
                          A−
                        </button>
                        <span
                          style={{
                            fontFamily: MONO,
                            fontSize: 11,
                            color: "var(--ink-soft)",
                            minWidth: 46,
                            textAlign: "center",
                            borderLeft: "1px solid var(--line)",
                            borderRight: "1px solid var(--line)",
                            padding: "6px 0",
                          }}
                        >
                          {Math.round(arScale * 100)}%
                        </span>
                        <button
                          onClick={() => setArScale((s) => Math.min(1.6, +(s + 0.1).toFixed(2)))}
                          style={{
                            border: "none",
                            background: "transparent",
                            cursor: "pointer",
                            padding: "6px 13px",
                            fontFamily: SERIF,
                            fontSize: 19,
                            color: "var(--brass)",
                            lineHeight: 1,
                          }}
                        >
                          A+
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* verse cards */}
                  {shown.map((r, i) => {
                    const ex = explanations[r.ref];
                    const tf = tafseers[r.ref];
                    const strength = maxScore > 0 ? r.score / maxScore : 0;
                    const diamonds = Math.max(1, Math.round(strength * 4));
                    return (
                      <motion.article
                        key={r.ref}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: Math.min(i, 10) * 0.04 }}
                        whileHover={{ y: -3, boxShadow: "0 34px 66px -42px var(--glow)" }}
                        style={{
                          background: "var(--surface)",
                          border: "1px solid var(--line)",
                          borderRadius: 20,
                          padding: "30px clamp(20px,3vw,34px)",
                          marginTop: 18,
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 12,
                            marginBottom: 20,
                            flexWrap: "wrap",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                            <span
                              style={{
                                fontFamily: MONO,
                                fontSize: "calc(13px * var(--ar-scale, 1))",
                                color: "var(--brass)",
                                border: "1px solid var(--line-strong)",
                                borderRadius: 999,
                                padding: ".32em .92em",
                                lineHeight: 1,
                              }}
                            >
                              [{r.ref}]
                            </span>
                            <span
                              style={{
                                fontFamily: SERIF,
                                fontStyle: "italic",
                                fontSize: "calc(15px * var(--ar-scale, 1))",
                                color: "var(--ink-soft)",
                              }}
                            >
                              {surahName(r.ref)}
                            </span>
                            <span style={{ ...monoEyebrow, fontSize: 9 }}>{matchLabel(r.matched)}</span>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                            <Diamonds n={diamonds} />
                            <span style={{ fontFamily: MONO, fontSize: 11, color: "var(--ink-faint)" }}>
                              {Math.round(strength * 100)}%
                            </span>
                          </div>
                        </div>

                        {r.arabic && (
                          <p
                            dir="rtl"
                            lang="ar"
                            style={{
                              fontFamily: "var(--ar-font, 'Amiri', serif)",
                              fontSize: "calc(34px * var(--ar-scale, 1))",
                              lineHeight: 2.05,
                              color: "var(--arabic)",
                              textAlign: "center",
                              margin: "0 0 20px",
                              padding: ".1em 0",
                            }}
                          >
                            {r.arabic}
                          </p>
                        )}
                        <p
                          style={{
                            fontFamily: SERIF,
                            fontSize: "calc(18px * var(--ar-scale, 1))",
                            lineHeight: 1.62,
                            color: "var(--ink)",
                            textAlign: "center",
                            maxWidth: "52ch",
                            margin: "0 auto 22px",
                          }}
                        >
                          {r.text}
                        </p>

                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: 10,
                            flexWrap: "wrap",
                          }}
                        >
                          {r.has_tafseer && (
                            <button
                              onClick={() => toggleTafseer(r.ref)}
                              disabled={tf?.loading}
                              className="dc-pill"
                              style={pillBtn}
                            >
                              {tf?.loading && <Loader2 className="animate-spin" size={12} />}
                              {tf && !tf.loading ? "Hide tafsīr" : "Show tafsīr"}
                            </button>
                          )}
                          {explainAvailable && (
                            <button
                              onClick={() => explain(r.ref)}
                              disabled={ex?.loading}
                              className="dc-pill"
                              style={pillBtn}
                            >
                              {ex?.loading && <Loader2 className="animate-spin" size={12} />}
                              {ex?.loading ? "Composing…" : ex ? "Hide explanation" : "Explain with AI"}
                            </button>
                          )}
                        </div>

                        <AnimatePresence>
                          {tf && !tf.loading && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              style={{ overflow: "hidden" }}
                            >
                              <div
                                style={{
                                  marginTop: 22,
                                  borderLeft: "2px solid var(--brass)",
                                  padding: "2px 0 2px 20px",
                                  textAlign: "left",
                                }}
                              >
                                <div
                                  style={{
                                    fontFamily: MONO,
                                    fontSize: 10,
                                    letterSpacing: ".16em",
                                    textTransform: "uppercase",
                                    color: "var(--brass)",
                                    marginBottom: 9,
                                  }}
                                >
                                  Tafsīr · source text
                                </div>
                                <p
                                  style={{
                                    fontFamily: SERIF,
                                    fontSize: "calc(16px * var(--ar-scale, 1))",
                                    lineHeight: 1.66,
                                    color: tf.error ? "var(--danger)" : "var(--ink-soft)",
                                    margin: 0,
                                    whiteSpace: "pre-wrap",
                                    maxHeight: 360,
                                    overflowY: "auto",
                                  }}
                                >
                                  {tf.error ?? tf.text}
                                </p>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>

                        <AnimatePresence>
                          {ex && !ex.loading && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              style={{ overflow: "hidden" }}
                            >
                              <div
                                style={{
                                  marginTop: 22,
                                  background: "var(--brass-soft)",
                                  border: "1px solid var(--line)",
                                  borderRadius: 14,
                                  padding: "18px 20px",
                                  textAlign: "left",
                                }}
                              >
                                <div
                                  style={{
                                    fontFamily: MONO,
                                    fontSize: 10,
                                    letterSpacing: ".14em",
                                    textTransform: "uppercase",
                                    color: "var(--brass)",
                                    marginBottom: 9,
                                  }}
                                >
                                  Composed · grounded in the tafsīr
                                </div>
                                <p
                                  style={{
                                    fontFamily: SERIF,
                                    fontSize: "calc(16px * var(--ar-scale, 1))",
                                    lineHeight: 1.68,
                                    color: ex.error ? "var(--danger)" : "var(--ink)",
                                    margin: 0,
                                    whiteSpace: "pre-wrap",
                                  }}
                                >
                                  {ex.error ?? ex.text}
                                </p>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.article>
                    );
                  })}

                  {ordered.length > visible && (
                    <button
                      onClick={() => setVisible((v) => v + PAGE)}
                      className="dc-pill"
                      style={{
                        ...pillBtn,
                        display: "block",
                        margin: "26px auto 0",
                        color: "var(--ink-soft)",
                        borderColor: "var(--line)",
                      }}
                    >
                      Show more · {ordered.length - visible} remaining
                    </button>
                  )}
                </>
              )}

              {/* relevance gate: topic not in the Qur'ān */}
              {resp && !resp.found && !resp.needs_confirmation && !searchError && (
                <div
                  style={{
                    marginTop: 30,
                    background: "var(--surface)",
                    border: "1px dashed var(--line-strong)",
                    borderRadius: 20,
                    padding: 34,
                    textAlign: "center",
                  }}
                >
                  <div style={{ ...monoEyebrow, marginBottom: 14 }}>You searched</div>
                  <div
                    style={{
                      fontFamily: MONO,
                      fontSize: 14,
                      color: "var(--ink-soft)",
                      border: "1px solid var(--line)",
                      borderRadius: 999,
                      display: "inline-block",
                      padding: "6px 16px",
                      marginBottom: 20,
                    }}
                  >
                    {asked}
                  </div>
                  <h3
                    style={{
                      fontFamily: SERIF,
                      fontWeight: 500,
                      fontSize: 26,
                      color: "var(--ink)",
                      margin: "0 0 10px",
                    }}
                  >
                    This topic isn't addressed in the Qur'ān.
                  </h3>
                  <p
                    style={{
                      fontFamily: SERIF,
                      fontSize: 16,
                      lineHeight: 1.6,
                      color: "var(--ink-soft)",
                      maxWidth: "44ch",
                      margin: "0 auto 16px",
                    }}
                  >
                    No verse cleared the relevance gate
                    {count ? ` across the ${count.toLocaleString()} indexed verses` : ""}. Rather than
                    force a weak match or invent an answer, the search declines — that honesty is the
                    core guarantee.
                  </p>
                  <span style={{ fontFamily: MONO, fontSize: 12, color: "var(--brass)" }}>
                    relevance_gate → no_hit
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ── Footer ──────────────────────────────────────────────── */}
          <div
            style={{
              borderTop: "1px solid var(--line)",
              padding: "24px clamp(18px,3vw,30px)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 16,
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontFamily: SERIF, fontSize: 14, color: "var(--ink-soft)" }}>
              Qur'ān RAG — a study &amp; search aid, not a substitute for scholarship.
            </span>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 10,
                letterSpacing: ".12em",
                textTransform: "uppercase",
                color: "var(--ink-faint)",
              }}
            >
              MIT · FastAPI · React
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
