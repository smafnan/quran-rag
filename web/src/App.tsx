import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, BookOpen, Loader2, Sparkles, MoonStar, ScrollText,
  ChevronDown, BookMarked, ListOrdered, Flame,
} from "lucide-react";

type Result = {
  ref: string;
  text: string;
  arabic: string;
  score: number;
  matched: string[];
  has_tafseer: boolean;
};
type Response = { found: boolean; source: string; count: number; results: Result[] };
type Explanation = { loading: boolean; text?: string; error?: string };
type TafseerPanel = { loading: boolean; text?: string; error?: string };

const SAMPLES = [
  "patience during hardship",
  "the story of Musa and Pharaoh",
  "forgiveness of sins",
  "the oneness of Allah",
];

const PAGE = 25;
const ARABIC_FONT = "'Amiri', 'Scheherazade New', 'Traditional Arabic', serif";

function refKey(ref: string): [number, number] {
  const [c, v] = ref.split(":").map(Number);
  return [c, v];
}

function Blobs() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden">
      <div className="absolute -top-48 left-1/4 h-[34rem] w-[34rem] rounded-full bg-emerald-600/20 blur-3xl animate-float" />
      <div className="absolute bottom-0 -right-40 h-[30rem] w-[30rem] rounded-full bg-teal-500/15 blur-3xl animate-float [animation-delay:-5s]" />
      <div className="absolute -bottom-40 -left-32 h-[28rem] w-[28rem] rounded-full bg-amber-500/10 blur-3xl animate-float [animation-delay:-9s]" />
    </div>
  );
}

function MatchBadges({ matched }: { matched: string[] }) {
  const label = matched.includes("keyword")
    ? { text: "word match", cls: "bg-emerald-400/15 text-emerald-200" }
    : matched.includes("semantic")
    ? { text: "related by meaning", cls: "bg-teal-400/15 text-teal-200" }
    : { text: "related", cls: "bg-white/10 text-emerald-200/60" };
  return (
    <span className={`rounded-md px-2 py-0.5 text-[10px] font-medium ${label.cls}`}>
      {label.text}
    </span>
  );
}

export default function App() {
  const [q, setQ] = useState("");
  const [asked, setAsked] = useState("");
  const [resp, setResp] = useState<Response | null>(null);
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState<number | null>(null);
  const [explainAvailable, setExplainAvailable] = useState(true);
  const [explanations, setExplanations] = useState<Record<string, Explanation>>({});
  const [tafseers, setTafseers] = useState<Record<string, TafseerPanel>>({});
  const [sortBy, setSortBy] = useState<"relevance" | "quran">("relevance");
  const [visible, setVisible] = useState(PAGE);
  const [searchError, setSearchError] = useState<string | null>(null);
  const reqSeq = useRef(0);

  useEffect(() => {
    fetch("/api/info")
      .then((r) => r.json())
      .then((d) => {
        setCount(d.passages);
        setExplainAvailable(!!d.explain_available);
      })
      .catch(() => {});
  }, []);

  async function ask(question?: string) {
    const text = (question ?? q).trim();
    if (!text || loading) return;
    const seq = ++reqSeq.current;
    setQ(text);
    setLoading(true);
    setExplanations({});
    setTafseers({});
    setVisible(PAGE);
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, mode: "all" }),
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
    } catch {
      if (seq === reqSeq.current) {
        setSearchError("Could not reach the search service.");
        setResp(null);
        setAsked(text);
      }
    } finally {
      if (seq === reqSeq.current) setLoading(false);
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
      const r = await fetch(`/api/tafseer?ref=${encodeURIComponent(ref)}`);
      if (!r.ok) throw new Error(`${r.status}`);
      const d = await r.json();
      setTafseers((prev) => ({
        ...prev,
        [ref]: {
          loading: false,
          text: d.tafseer || "No tafseer is available for this verse in the loaded corpus.",
        },
      }));
    } catch {
      setTafseers((prev) => ({
        ...prev,
        [ref]: { loading: false, error: "Could not load the tafseer." },
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
      const r = await fetch("/api/explain", {
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

  const shown = ordered.slice(0, visible);

  return (
    <div className="relative min-h-screen text-emerald-50">
      <Blobs />
      <div className="relative mx-auto max-w-3xl px-5 py-14">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10 text-center"
        >
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/5 px-4 py-1.5 text-xs text-emerald-200/80 backdrop-blur">
            <MoonStar size={14} className="text-amber-300" />
            Every occurrence, strictly from the Quran
          </div>
          <h1 className="font-serif text-5xl font-bold tracking-tight text-white sm:text-6xl">
            Quran{" "}
            <span className="bg-gradient-to-r from-emerald-300 via-teal-200 to-amber-200 bg-clip-text text-transparent">
              Search
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-emerald-200/60">
            Enter a topic — see every verse connected to it across the whole
            Quran, in Arabic and English, with the tafseer one click away.
          </p>
        </motion.header>

        {/* Search box */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl border border-emerald-400/15 bg-white/[0.04] p-2 shadow-2xl shadow-emerald-900/40 backdrop-blur"
        >
          <div className="flex items-center gap-2">
            <Search className="ml-3 text-emerald-300/70" size={20} />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
              placeholder="Enter a topic — e.g. patience, Musa, mercy…"
              className="flex-1 bg-transparent px-2 py-3 text-base text-white placeholder:text-emerald-200/30 outline-none"
            />
            <button
              onClick={() => ask()}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-500/20 disabled:opacity-60"
            >
              {loading ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
              Seek
            </button>
          </div>
        </motion.div>

        {/* Sample chips */}
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {SAMPLES.map((s) => (
            <button
              key={s}
              onClick={() => ask(s)}
              className="rounded-full border border-emerald-400/15 bg-emerald-400/5 px-3 py-1 text-xs text-emerald-200/70 hover:bg-emerald-400/10"
            >
              {s}
            </button>
          ))}
        </div>

        {/* Search error (rate limit, upstream failure) */}
        {searchError && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-9 rounded-2xl border border-rose-400/25 bg-rose-400/5 p-6 text-center"
          >
            <p className="text-rose-100/90">{searchError}</p>
            <p className="mt-2 text-sm text-rose-200/50">
              This is a limit or a connection problem — not a statement about the text.
            </p>
          </motion.div>
        )}

        {/* Results */}
        <AnimatePresence mode="wait">
          {resp && (
            <motion.div
              key={asked}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-9 space-y-4"
            >
              {!resp.found ? (
                <div className="rounded-2xl border border-amber-400/20 bg-amber-400/5 p-8 text-center">
                  <BookOpen className="mx-auto mb-3 text-amber-300/70" size={28} />
                  <p className="text-amber-100/90">
                    This topic does not appear to be addressed in the Quran
                    {count ? ` (within the ${count} indexed verses).` : "."}
                  </p>
                  <p className="mt-2 text-sm text-amber-200/50">
                    The system never answers beyond the text.
                  </p>
                </div>
              ) : (
                <>
                  {/* Results toolbar */}
                  <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-400/10 bg-white/[0.03] px-4 py-2.5">
                    <span className="inline-flex items-center gap-2 text-sm text-emerald-100/90">
                      <Flame size={15} className="text-amber-300" />
                      <strong>{resp.count}</strong>&nbsp;verse{resp.count === 1 ? "" : "s"} connected to
                      “{asked}”
                    </span>
                    <div className="flex items-center gap-1 text-xs">
                      <button
                        onClick={() => setSortBy("relevance")}
                        className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 ${
                          sortBy === "relevance"
                            ? "bg-emerald-400/15 text-emerald-100"
                            : "text-emerald-200/50 hover:bg-white/5"
                        }`}
                      >
                        <Sparkles size={12} /> Relevance
                      </button>
                      <button
                        onClick={() => setSortBy("quran")}
                        className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 ${
                          sortBy === "quran"
                            ? "bg-emerald-400/15 text-emerald-100"
                            : "text-emerald-200/50 hover:bg-white/5"
                        }`}
                      >
                        <ListOrdered size={12} /> Quran order
                      </button>
                    </div>
                  </div>

                  {shown.map((r, i) => {
                    const ex = explanations[r.ref];
                    const tf = tafseers[r.ref];
                    return (
                      <motion.article
                        key={r.ref}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: Math.min(i, 10) * 0.04 }}
                        className="rounded-2xl border border-emerald-400/12 bg-white/[0.04] p-6 backdrop-blur"
                      >
                        <div className="mb-3 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="rounded-lg bg-emerald-400/15 px-3 py-1 font-mono text-sm font-semibold text-emerald-200">
                              {r.ref}
                            </span>
                            <MatchBadges matched={r.matched} />
                          </div>
                          <span className="text-[11px] text-emerald-200/40">
                            {r.score.toFixed(2)}
                          </span>
                        </div>

                        {r.arabic && (
                          <p
                            dir="rtl"
                            lang="ar"
                            className="mb-3 text-right text-2xl leading-loose text-amber-50/95"
                            style={{ fontFamily: ARABIC_FONT }}
                          >
                            {r.arabic}
                          </p>
                        )}
                        <p className="font-serif text-lg leading-relaxed text-emerald-50/95">
                          {r.text}
                        </p>

                        <div className="mt-4 flex flex-wrap gap-2">
                          {r.has_tafseer && (
                            <button
                              onClick={() => toggleTafseer(r.ref)}
                              disabled={tf?.loading}
                              className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-300/20 bg-emerald-300/5 px-3 py-1.5 text-xs font-medium text-emerald-200/80 transition hover:bg-emerald-300/10 disabled:opacity-60"
                            >
                              {tf?.loading ? (
                                <Loader2 className="animate-spin" size={13} />
                              ) : (
                                <BookMarked size={13} />
                              )}
                              {tf && !tf.loading ? "Hide tafseer" : "Show tafseer"}
                              {!tf?.loading && (
                                <ChevronDown
                                  size={13}
                                  className={`transition-transform ${tf ? "rotate-180" : ""}`}
                                />
                              )}
                            </button>
                          )}
                          {explainAvailable && (
                            <button
                              onClick={() => explain(r.ref)}
                              disabled={ex?.loading}
                              className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300/20 bg-amber-300/5 px-3 py-1.5 text-xs font-medium text-amber-200/80 transition hover:bg-amber-300/10 disabled:opacity-60"
                            >
                              {ex?.loading ? (
                                <Loader2 className="animate-spin" size={13} />
                              ) : (
                                <ScrollText size={13} />
                              )}
                              {ex?.loading
                                ? "Consulting the tafseer…"
                                : ex
                                ? "Hide explanation"
                                : "Explain with AI"}
                            </button>
                          )}
                        </div>

                        <AnimatePresence>
                          {tf && !tf.loading && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className="overflow-hidden"
                            >
                              <div className="mt-4 max-h-80 overflow-y-auto rounded-xl border border-emerald-300/15 bg-emerald-300/[0.04] p-5">
                                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-emerald-300/50">
                                  Tafseer (source text)
                                </p>
                                {tf.error ? (
                                  <p className="text-sm text-emerald-200/70">{tf.error}</p>
                                ) : (
                                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-emerald-50/80">
                                    {tf.text}
                                  </p>
                                )}
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
                              className="overflow-hidden"
                            >
                              <div className="mt-4 rounded-xl border border-amber-300/15 bg-amber-300/[0.04] p-5">
                                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-amber-300/50">
                                  AI explanation (grounded in the tafseer)
                                </p>
                                {ex.error ? (
                                  <p className="text-sm text-amber-200/70">{ex.error}</p>
                                ) : (
                                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-emerald-50/85">
                                    {ex.text}
                                  </p>
                                )}
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
                      className="mx-auto block rounded-xl border border-emerald-400/15 bg-emerald-400/5 px-6 py-2.5 text-sm text-emerald-200/80 hover:bg-emerald-400/10"
                    >
                      Show more ({ordered.length - visible} remaining)
                    </button>
                  )}
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="mt-16 text-center text-xs text-emerald-200/30">
          Hybrid search: exact words + meaning · Arabic & English · tafseer on click
        </footer>
      </div>
    </div>
  );
}
