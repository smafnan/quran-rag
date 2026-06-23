import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, BookOpen, Loader2, Sparkles, MoonStar } from "lucide-react";

type Result = { ref: string; text: string; score: number };
type Response = { found: boolean; source: string; results: Result[] };

const SAMPLES = [
  "What does it say about hardship and ease?",
  "The oneness of God",
  "Remembrance and the heart",
  "Patience and good deeds",
];

function Blobs() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden">
      <div className="absolute -top-48 left-1/4 h-[34rem] w-[34rem] rounded-full bg-emerald-600/20 blur-3xl animate-float" />
      <div className="absolute bottom-0 -right-40 h-[30rem] w-[30rem] rounded-full bg-teal-500/15 blur-3xl animate-float [animation-delay:-5s]" />
      <div className="absolute -bottom-40 -left-32 h-[28rem] w-[28rem] rounded-full bg-amber-500/10 blur-3xl animate-float [animation-delay:-9s]" />
    </div>
  );
}

export default function App() {
  const [q, setQ] = useState("");
  const [resp, setResp] = useState<Response | null>(null);
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/info").then((r) => r.json()).then((d) => setCount(d.passages)).catch(() => {});
  }, []);

  async function ask(question?: string) {
    const text = (question ?? q).trim();
    if (!text) return;
    setQ(text);
    setLoading(true);
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, k: 5 }),
      });
      setResp(await r.json());
    } finally {
      setLoading(false);
    }
  }

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
            Answers strictly and only from the Quran
          </div>
          <h1 className="font-serif text-5xl font-bold tracking-tight text-white sm:text-6xl">
            Quran{" "}
            <span className="bg-gradient-to-r from-emerald-300 via-teal-200 to-amber-200 bg-clip-text text-transparent">
              Search
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-emerald-200/60">
            Ask anything. Every answer is drawn directly from the verses, always
            cited — and if the text doesn’t address it, it says so.
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
              placeholder="Ask the Quran…"
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

        {/* Results */}
        <AnimatePresence mode="wait">
          {resp && (
            <motion.div
              key={q}
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
                resp.results.map((r, i) => (
                  <motion.article
                    key={r.ref}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.06 }}
                    className="rounded-2xl border border-emerald-400/12 bg-white/[0.04] p-6 backdrop-blur"
                  >
                    <div className="mb-3 flex items-center justify-between">
                      <span className="rounded-lg bg-emerald-400/15 px-3 py-1 font-mono text-sm font-semibold text-emerald-200">
                        {r.ref}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-amber-300"
                            style={{ width: `${Math.min(100, r.score * 140)}%` }}
                          />
                        </div>
                        <span className="text-[11px] text-emerald-200/40">
                          {r.score.toFixed(2)}
                        </span>
                      </div>
                    </div>
                    <p className="font-serif text-lg leading-relaxed text-emerald-50/95">
                      {r.text}
                    </p>
                  </motion.article>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="mt-16 text-center text-xs text-emerald-200/30">
          Ships with a small sample · replace data/quran_sample.jsonl with the full
          translation you trust · grounded RAG, verse citations
        </footer>
      </div>
    </div>
  );
}
