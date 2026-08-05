"""
Diversity measurement and filtering for DrBencher.

Two modes:
  - Online filter: during generation, reject near-duplicate candidates
  - Post-hoc measurement: score a finished benchmark on diversity metrics

Metrics: Vendi Score, Self-BLEU, near-duplicate rate, pairwise distance stats,
tag entropy, subarea Gini coefficient, MMD (inter-benchmark).

Run via:  python -m drbench.diversity --benchmark_file bench.json [--output report.json]
"""

import argparse
import json
import logging
import math
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guards (same pattern as coding_challenge_harmony.py)
# ---------------------------------------------------------------------------

# sentence-transformers pulls the torch stack, so we NEVER import it at module
# load — it is loaded lazily in _load_sbert() only when embeddings are actually
# requested. DRBENCHER_NO_TORCH=1 disables it entirely (used by the serve client
# on a login node that kills torch-importing processes); the TF-IDF fallback
# below is torch-free.
_NO_TORCH = os.environ.get("DRBENCHER_NO_TORCH") == "1"


def _load_sbert(model_name):
    """Lazily load a SentenceTransformer, or return None if disabled/unavailable.

    Not just ImportError-safe: sentence-transformers can pull optional multimodal
    backends (torchcodec/torchaudio, needing system FFmpeg) that fail to LOAD with
    OSError/RuntimeError. Any failure -> None -> non-embedding fallback.
    """
    if _NO_TORCH:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning("SentenceTransformer unavailable (%s); using TF-IDF fallback", e)
        return None

try:
    from scipy.linalg import eigvalsh as scipy_eigvalsh
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================================
# Text extraction
# ============================================================================

def extract_text(problem: dict, text_keys: Optional[List[str]] = None) -> str:
    """Extract text from a benchmark problem dict.

    If *text_keys* is given, concatenate values from those keys.
    Otherwise try ``question`` -> ``problem_statement`` -> ``name``.
    """
    if text_keys:
        parts = [str(problem[k]) for k in text_keys if k in problem]
        return " ".join(parts)
    for key in ("question", "problem_statement", "name"):
        if key in problem and problem[key]:
            return str(problem[key])
    return ""


# ============================================================================
# Embedding
# ============================================================================

def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """Embed a list of texts into L2-normalized vectors.

    Primary: sentence-transformers.  Fallback: TF-IDF + TruncatedSVD.
    Optionally loads/saves a ``.npy`` cache.
    """
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            logger.info("Loaded cached embeddings from %s", cache_path)
            return cached
        logger.warning(
            "Cache shape mismatch (%d vs %d); re-embedding.",
            cached.shape[0], len(texts),
        )

    model = _load_sbert(model_name)
    if model is not None:
        logger.info("Embedding %d texts with SentenceTransformer(%s)", len(texts), model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
    elif _HAS_SKLEARN:
        logger.info("Embedding %d texts with TF-IDF + TruncatedSVD fallback", len(texts))
        vectorizer = TfidfVectorizer(max_features=10000)
        try:
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary (e.g., all stop words / single chars)
            logger.warning("TF-IDF produced empty vocabulary; using random embeddings")
            rng = np.random.RandomState(42)
            embeddings = rng.randn(len(texts), 64).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.save(cache_path, embeddings)
            return embeddings
        n_features = tfidf.shape[1]
        n_components = min(384, len(texts) - 1, n_features)
        svd = TruncatedSVD(n_components=max(n_components, 1))
        embeddings = svd.fit_transform(tfidf).astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms
    else:
        raise RuntimeError(
            "Neither sentence-transformers nor scikit-learn is installed. "
            "Install at least one for embedding support."
        )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Saved embeddings cache to %s", cache_path)

    return embeddings


def embed_problems(
    problems: List[dict],
    text_keys: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """Convenience: extract text from each problem then embed."""
    texts = [extract_text(p, text_keys) for p in problems]
    return embed_texts(texts, model_name=model_name, cache_path=cache_path)


# ============================================================================
# Diversity metrics
# ============================================================================

def vendi_score(embeddings: np.ndarray) -> float:
    """Vendi Score: effective number of unique items via eigenvalue entropy.

    Uses cosine similarity kernel (embeddings must be L2-normalized).
    """
    n = embeddings.shape[0]
    if n <= 1:
        return float(n)

    K = embeddings @ embeddings.T
    if _HAS_SCIPY:
        eigs = scipy_eigvalsh(K)
    else:
        eigs = np.linalg.eigvalsh(K)

    # Normalize eigenvalues to form a probability distribution
    eigs = np.real(eigs)
    eigs = eigs / eigs.sum()
    # Filter out non-positive values for log safety
    eigs = eigs[eigs > 0]
    entropy = -np.sum(eigs * np.log(eigs))
    return float(np.exp(entropy))


def _count_ngrams(tokens: List[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _sentence_bleu(hypothesis_tokens: List[str], references_tokens: List[List[str]], max_n: int = 4) -> float:
    """Compute sentence-level BLEU (no smoothing) for a single hypothesis."""
    if not hypothesis_tokens:
        return 0.0

    log_bleu = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _count_ngrams(hypothesis_tokens, n)
        if not hyp_ngrams:
            return 0.0
        # Clipped counts: for each n-gram, max count across references
        clipped = Counter()
        for ref_tokens in references_tokens:
            ref_ngrams = _count_ngrams(ref_tokens, n)
            for ng in hyp_ngrams:
                clipped[ng] = max(clipped[ng], ref_ngrams[ng])
        numerator = sum(min(hyp_ngrams[ng], clipped[ng]) for ng in hyp_ngrams)
        denominator = sum(hyp_ngrams.values())
        if numerator == 0:
            return 0.0
        log_bleu += math.log(numerator / denominator)

    log_bleu /= max_n

    # Brevity penalty
    hyp_len = len(hypothesis_tokens)
    closest_ref_len = min(
        (abs(len(r) - hyp_len), len(r)) for r in references_tokens
    )[1]
    if hyp_len < closest_ref_len:
        log_bleu += 1.0 - closest_ref_len / hyp_len

    return math.exp(log_bleu)


def self_bleu(texts: List[str], n: int = 4) -> float:
    """Self-BLEU: average BLEU of each text against all others.

    Lower is more diverse.  Uses simple whitespace tokenization.
    """
    if len(texts) < 2:
        return 0.0

    MAX_TEXTS = 1000
    if len(texts) > MAX_TEXTS:
        warnings.warn(
            f"self_bleu: subsampling {MAX_TEXTS} of {len(texts)} texts for efficiency"
        )
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), MAX_TEXTS, replace=False)
        texts = [texts[i] for i in indices]

    tokenized = [t.lower().split() for t in texts]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        scores.append(_sentence_bleu(hyp, refs, max_n=n))

    return float(np.mean(scores))


def near_duplicate_rate(embeddings: np.ndarray, threshold: float = 0.95) -> float:
    """Fraction of pairs with cosine similarity above *threshold*."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    sim = embeddings @ embeddings.T
    # Upper triangle (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    upper = sim[iu]
    return float((upper > threshold).sum() / len(upper))


def pairwise_distance_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Pairwise cosine distance stats: mean, median, min, max, std."""
    n = embeddings.shape[0]
    if n < 2:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    sim = embeddings @ embeddings.T
    iu = np.triu_indices(n, k=1)
    dists = 1.0 - sim[iu]
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "std": float(np.std(dists)),
    }


def tag_entropy(problems: List[dict], tag_key: str = "tags") -> float:
    """Shannon entropy of the tag frequency distribution."""
    all_tags: List[str] = []
    for p in problems:
        tags = p.get(tag_key, [])
        if isinstance(tags, list):
            all_tags.extend(tags)
        elif isinstance(tags, str):
            all_tags.append(tags)
    if not all_tags:
        return 0.0
    counts = Counter(all_tags)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs)))


def subarea_gini(problems: List[dict], key: str = "subarea") -> float:
    """Gini coefficient of problem counts per subarea.

    0 = perfect equality, 1 = maximal inequality.
    """
    counts = Counter(p.get(key, "unknown") for p in problems)
    if not counts:
        return 0.0
    values = np.array(sorted(counts.values()), dtype=np.float64)
    n = len(values)
    if n == 1:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def mmd(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    kernel: str = "cosine",
) -> float:
    """Maximum Mean Discrepancy between two embedding sets.

    MMD^2 = mean(K_aa) + mean(K_bb) - 2*mean(K_ab).
    Kernels: ``"cosine"`` (dot product) or ``"rbf"`` (median heuristic bandwidth).
    """
    if kernel == "cosine":
        k_aa = embeddings_a @ embeddings_a.T
        k_bb = embeddings_b @ embeddings_b.T
        k_ab = embeddings_a @ embeddings_b.T
    elif kernel == "rbf":
        from scipy.spatial.distance import cdist
        d_aa = cdist(embeddings_a, embeddings_a, metric="sqeuclidean")
        d_bb = cdist(embeddings_b, embeddings_b, metric="sqeuclidean")
        d_ab = cdist(embeddings_a, embeddings_b, metric="sqeuclidean")
        # Median heuristic
        all_dists = np.concatenate([d_aa.ravel(), d_bb.ravel(), d_ab.ravel()])
        sigma2 = float(np.median(all_dists))
        if sigma2 == 0:
            sigma2 = 1.0
        k_aa = np.exp(-d_aa / (2 * sigma2))
        k_bb = np.exp(-d_bb / (2 * sigma2))
        k_ab = np.exp(-d_ab / (2 * sigma2))
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    mmd_sq = float(k_aa.mean() + k_bb.mean() - 2 * k_ab.mean())
    return max(mmd_sq, 0.0)


# ============================================================================
# Diversity filter (online)
# ============================================================================

def diversity_filter(
    candidates: List[dict],
    target_count: int,
    min_distance: float = 0.05,
    text_keys: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple:
    """Greedy max-min distance selection for diverse subset.

    Returns ``(selected, rejected)`` — two disjoint lists whose union equals
    *candidates*.

    1. Embed all candidates.
    2. Seed with first candidate.
    3. Incrementally track max similarity to selected set.
    4. Pick candidate that maximizes min-distance and meets threshold.
    """
    if len(candidates) <= 1:
        return list(candidates), []

    embeddings = embed_problems(candidates, text_keys=text_keys, model_name=model_name)
    n = len(candidates)

    selected_idx = [0]
    # max_sim_to_selected[i] = max cosine sim of candidate i to any selected candidate
    max_sim = embeddings @ embeddings[0:1].T
    max_sim = max_sim.squeeze(1)  # shape (n,)

    while len(selected_idx) < target_count:
        # Mask already selected
        mask = np.ones(n, dtype=bool)
        mask[selected_idx] = False

        distances = 1.0 - max_sim
        # Among unselected, find candidates meeting threshold
        valid = mask & (distances >= min_distance)
        if not valid.any():
            # No candidate meets min_distance — stop selecting
            break

        # Pick the one with largest min-distance
        best = np.argmax(np.where(valid, distances, -np.inf))
        selected_idx.append(int(best))

        # Update max_sim incrementally
        new_sim = embeddings @ embeddings[best:best + 1].T
        new_sim = new_sim.squeeze(1)
        max_sim = np.maximum(max_sim, new_sim)

    selected_set = set(selected_idx)
    selected = [candidates[i] for i in selected_idx]
    rejected = [candidates[i] for i in range(n) if i not in selected_set]
    return selected, rejected


class GraphDiversityFilter:
    """Graph-based diversity filter using maximum independent set approximation.

    Builds a near-duplicate graph (edge between i and j when their cosine
    distance < *min_distance*) then iteratively removes the node with the
    highest degree.  The surviving nodes form an approximate maximum
    independent set — the largest subset where every pair is diverse.

    Unlike the greedy selector this has **no seed bias** and tends to
    **maximise the number of kept items**.
    """

    def __init__(
        self,
        min_distance: float = 0.05,
        text_keys: Optional[List[str]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.min_distance = min_distance
        self.text_keys = text_keys
        self.model_name = model_name

    def filter(self, candidates: List[dict]) -> tuple:
        """Return ``(selected, rejected)``."""
        if len(candidates) <= 1:
            return list(candidates), []

        embeddings = embed_problems(
            candidates, text_keys=self.text_keys, model_name=self.model_name,
        )
        n = len(candidates)

        # Pairwise cosine similarity → near-duplicate adjacency
        sim = embeddings @ embeddings.T
        threshold = 1.0 - self.min_distance
        adj = sim > threshold
        np.fill_diagonal(adj, False)

        degree = adj.sum(axis=1).copy()
        rejected_idx = set()

        while degree.max() > 0:
            # Remove the node with the most near-duplicate neighbours.
            # Tie-break: highest total similarity to neighbours (most redundant).
            max_deg = degree.max()
            tied = np.where(degree == max_deg)[0]
            if len(tied) == 1:
                worst = int(tied[0])
            else:
                worst_scores = [float(sim[i][adj[i]].sum()) for i in tied]
                worst = int(tied[np.argmax(worst_scores)])

            rejected_idx.add(worst)

            # Remove worst from the graph
            neighbours = np.where(adj[worst])[0]
            adj[worst, :] = False
            adj[:, worst] = False
            degree[worst] = 0
            for nb in neighbours:
                degree[nb] = int(adj[nb].sum())

        selected = [candidates[i] for i in range(n) if i not in rejected_idx]
        rejected = [candidates[i] for i in range(n) if i in rejected_idx]
        return selected, rejected


# ============================================================================
# Reporting
# ============================================================================

def diversity_report(
    problems: List[dict],
    text_keys: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_path: Optional[str] = None,
) -> dict:
    """Compute all diversity metrics and print a formatted summary."""
    embeddings = embed_problems(problems, text_keys=text_keys, model_name=model_name, cache_path=cache_path)
    texts = [extract_text(p, text_keys) for p in problems]

    vs = vendi_score(embeddings)
    sb = self_bleu(texts)
    ndr = near_duplicate_rate(embeddings)
    pd_stats = pairwise_distance_stats(embeddings)
    te = tag_entropy(problems)
    sg = subarea_gini(problems)

    report = {
        "num_problems": len(problems),
        "vendi_score": vs,
        "self_bleu_4gram": sb,
        "near_dup_rate": ndr,
        "pairwise_distance": pd_stats,
        "tag_entropy": te,
        "subarea_gini": sg,
    }

    print("=== Diversity Report ===")
    print(f"Problems:            {len(problems)}")
    print(f"Vendi Score:         {vs:.2f}")
    print(f"Self-BLEU (4-gram):  {sb:.4f}")
    print(f"Near-dup rate:       {ndr:.4f}")
    print(f"Pairwise distance:   mean={pd_stats['mean']:.4f} median={pd_stats['median']:.4f}")
    print(f"Tag entropy:         {te:.4f}")
    print(f"Subarea Gini:        {sg:.4f}")
    print("========================")

    return report


def cross_benchmark_diversity(
    benchmarks: Dict[str, List[dict]],
    text_keys: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """Compute pairwise MMD between benchmarks and per-benchmark diversity."""
    names = list(benchmarks.keys())
    all_embeddings = {}
    per_report = {}

    # Embed all benchmarks together so dimensions are consistent (TF-IDF fallback)
    all_texts = []
    slices = {}
    for name in names:
        start = len(all_texts)
        texts = [extract_text(p, text_keys) for p in benchmarks[name]]
        all_texts.extend(texts)
        slices[name] = (start, start + len(texts))
    combined_emb = embed_texts(all_texts, model_name=model_name)
    for name in names:
        s, e = slices[name]
        all_embeddings[name] = combined_emb[s:e]
        per_report[name] = diversity_report(benchmarks[name], text_keys=text_keys, model_name=model_name)

    n = len(names)
    mmd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = mmd(all_embeddings[names[i]], all_embeddings[names[j]])
            mmd_matrix[i, j] = m
            mmd_matrix[j, i] = m

    # Print MMD table
    print("\n=== Cross-Benchmark MMD ===")
    header = "".ljust(20) + "".join(nm[:12].ljust(14) for nm in names)
    print(header)
    for i, nm in enumerate(names):
        row = nm[:20].ljust(20) + "".join(f"{mmd_matrix[i, j]:.4f}".ljust(14) for j in range(n))
        print(row)
    print("===========================")

    return {
        "mmd_matrix": mmd_matrix.tolist(),
        "benchmark_names": names,
        "per_benchmark_reports": per_report,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diversity measurement and filtering")
    parser.add_argument("--benchmark_file", required=True, help="Path to benchmark JSON file")
    parser.add_argument("--text_keys", nargs="*", default=None, help="Keys to extract text from")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Embedding model name")
    parser.add_argument("--compare_with", default=None, help="Comma-separated paths to other benchmark files for cross-benchmark MMD")
    parser.add_argument("--output", default=None, help="Path to save report JSON")
    parser.add_argument("--filter", action="store_true", help="Run diversity filter instead of reporting")
    parser.add_argument("--method", choices=["greedy", "graph"], default="graph",
                        help="Filter method: graph (max independent set) or greedy (max-min distance)")
    parser.add_argument("--target_count", type=int, default=100, help="Target count for greedy filter")
    parser.add_argument("--min_distance", type=float, default=0.05, help="Min cosine distance for filter")
    args = parser.parse_args()

    with open(args.benchmark_file) as f:
        problems = json.load(f)
    if isinstance(problems, dict):
        # Some bench files store problems under a key
        for key in ("problems", "questions", "data"):
            if key in problems:
                problems = problems[key]
                break

    if args.filter:
        print(f"Method: {args.method} | min_distance: {args.min_distance}")
        if args.method == "graph":
            gf = GraphDiversityFilter(
                min_distance=args.min_distance,
                text_keys=args.text_keys,
                model_name=args.model_name,
            )
            selected, rejected = gf.filter(problems)
        else:
            selected, rejected = diversity_filter(
                problems,
                target_count=args.target_count,
                min_distance=args.min_distance,
                text_keys=args.text_keys,
                model_name=args.model_name,
            )
        print(f"Selected {len(selected)} / {len(problems)} problems ({len(rejected)} rejected)")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(selected, f, indent=2)
            print(f"Saved filtered set to {args.output}")
            if rejected:
                bad_path = args.output.replace(".json", "_bad.json")
                with open(bad_path, "w") as f:
                    json.dump(rejected, f, indent=2)
                print(f"Saved {len(rejected)} rejected samples to {bad_path}")
        # Also print a quick report on the filtered set
        diversity_report(selected, text_keys=args.text_keys, model_name=args.model_name)
    elif args.compare_with:
        benchmarks = {"main": problems}
        for path in args.compare_with.split(","):
            path = path.strip()
            with open(path) as f:
                other = json.load(f)
            if isinstance(other, dict):
                for key in ("problems", "questions", "data"):
                    if key in other:
                        other = other[key]
                        break
            label = os.path.splitext(os.path.basename(path))[0]
            benchmarks[label] = other
        result = cross_benchmark_diversity(benchmarks, text_keys=args.text_keys, model_name=args.model_name)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved cross-benchmark report to {args.output}")
    else:
        report = diversity_report(problems, text_keys=args.text_keys, model_name=args.model_name)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
