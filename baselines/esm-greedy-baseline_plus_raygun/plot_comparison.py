import parasail
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
from Bio import SeqIO

# ── File paths ──────────────────────────────────────────────────────────────
baseline_rec = {
    "mcherry": "results/mcherry-esm2-baseline-full/esm2-mcherry-full-180-200_10-full.fasta",
    "egfp":    "results/egfp-baseline-full-run/egfp-full-180-200_10-full.fasta",
    "ras":     "results/ras-baseline-esm2-150-160/None-150-160_10-full.fasta",
}
base_esmc_rec = {
    "mcherry": "results/mcherry-esmc-baseline-full/esmc-mcherry-full-180-200_10-full.fasta",
    "egfp":    "results/egfp-baseline-esmc-full-run/esmc-full-run-180-200_10-full.fasta",
    "ras":     "results/ras-baseline-esmc-v2-150-160/None-150-160_10-full.fasta",
}
raygun_rec = {
    "mcherry": "results/mcherry-raygun-v1-results/unfiltered_mcherry_0.01_1000.fasta",
    "egfp":    "results/egfp-raygun-v1-results/unfiltered_egfp_0.05_1000.fasta",
    "ras":     "results/ras-raygun-v1-results-150-160/unfiltered_ras_0.05_1000.fasta",
}
raygun_rec_50 = {
    "mcherry": "results/mcherry-raygun-v1-results-noise-0.5/unfiltered_mcherry_0.5_1000.fasta",
    "egfp":    "results/egfp-raygun-v1-results-noise-0.5/unfiltered_egfp_0.5_1000.fasta",
    "ras":     "results/ras-raygun-v1-results-150-160-noise-0.5/unfiltered_ras_0.5_1000.fasta",
}
fastafiles = {
    "mcherry": "results/mcherry.fasta",
    "egfp":    "results/egfp.fasta",
    "ras":     "results/ras.fasta",
}
importantloc = {
    "egfp":    [66, 67, 68],
    "mcherry": [71, 72, 73],
    "ras":     [10, 15, 16, 17],
}

# ── Alignment helpers ────────────────────────────────────────────────────────
def get_true_index(aligned, unaligned_idx):
    aps = [(x, idx) for x, idx in zip(aligned, range(len(aligned))) if x != "-"]
    return aps[unaligned_idx]

def compute_alignment(seq, seqt, pos_to_query):
    res = parasail.nw_trace_scan_16(seq, seqt, 1, 1, parasail.blosum62)
    query_ = ""
    ref_   = ""
    for p in pos_to_query:
        x_, idx_ = get_true_index(res.traceback.query, p - 1)
        query_  += x_
        ref_    += res.traceback.ref[idx_]
    return query_ == ref_

def scoring(sc1, sc2):
    s1 = parasail.nw_stats_diag_16(sc1, sc2, 1, 1, parasail.blosum62)
    return s1.matches / s1.length

def compute_score(seq, recs, loc):
    sc, sq = [], []
    for rec in recs:
        sc.append(compute_alignment(seq, str(rec.seq), pos_to_query=loc))
        sq.append(scoring(seq, str(rec.seq)))
    return np.mean(sc), np.mean(sq)

def get_recs_seqs(prot):
    trec_b   = list(SeqIO.parse(baseline_rec[prot],   "fasta"))
    trec_e   = list(SeqIO.parse(base_esmc_rec[prot],  "fasta"))
    trec_r   = list(SeqIO.parse(raygun_rec[prot],     "fasta"))
    trec_r50 = list(SeqIO.parse(raygun_rec_50[prot],  "fasta"))
    seq      = str(SeqIO.read(fastafiles[prot], "fasta").seq)
    loc      = importantloc[prot]
    maxrec   = min(len(trec_b), len(trec_e), len(trec_r50), len(trec_r))
    return seq, trec_b[:maxrec], trec_e[:maxrec], trec_r[:maxrec], trec_r50[:maxrec], loc

# ── Compute scores ───────────────────────────────────────────────────────────
res_seqid = []
res_score = []
for p in ["egfp", "mcherry", "ras"]:
    seq, tb, te, tr, tr50, loc = get_recs_seqs(p)
    esm2score, esm2seqid = compute_score(seq, tb,   loc)
    esmcscore, esmcseqid = compute_score(seq, te,   loc)
    rayscore,  rayseqid  = compute_score(seq, tr,   loc)
    ray50sc,   ray50sqid = compute_score(seq, tr50, loc)
    res_seqid += [
        [p, "Raygun (noise=0.05)", rayseqid],
        [p, "Raygun (noise=0.5)",  ray50sqid],
        [p, "ESM-2 PLL",          esm2seqid],
        [p, "ESM-C PLL",          esmcseqid],
    ]
    res_score += [
        [p, "Raygun (noise=0.05)", rayscore],
        [p, "Raygun (noise=0.5)",  ray50sc],
        [p, "ESM-2 PLL",          esm2score],
        [p, "ESM-C PLL",          esmcscore],
    ]

dfseqid    = pd.DataFrame(res_seqid, columns=["protein", "model", "score"])
dfretention = pd.DataFrame(res_score, columns=["protein", "model", "score"])

# ── Nature journal style ─────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "svg.fonttype":       "none",
    "font.family":        "Arial",
    "font.size":          7,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.0,
    "ytick.major.size":   2.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.bottom":       False,
})

# ── Colours – Wong (2011) colorblind-safe palette ────────────────────────────
PALETTE = {
    "Raygun (σ=0.05)": "#0072B2",
    "Raygun (σ=0.5)":  "#56B4E9",
    "ESM-C":           "#D55E00",
    "ESM-2":           "#E69F00",
}
MODEL_ORDER  = ["Raygun (σ=0.05)", "Raygun (σ=0.5)", "ESM-C", "ESM-2"]
MODEL_RENAME = {
    "Raygun (noise=0.05)": "Raygun (σ=0.05)",
    "Raygun (noise=0.5)":  "Raygun (σ=0.5)",
    "ESM-C PLL":           "ESM-C",
    "ESM-2 PLL":           "ESM-2",
}
PROT_RENAME = {"egfp": "eGFP", "mcherry": "mCherry", "ras": "KRAS"}
PROT_ORDER  = ["eGFP", "mCherry", "KRAS"]

def prep(df):
    out = df.copy()
    out["model"]   = pd.Categorical(
        out["model"].astype(str).map(MODEL_RENAME),
        categories=MODEL_ORDER, ordered=True)
    out["protein"] = pd.Categorical(
        out["protein"].map(PROT_RENAME),
        categories=PROT_ORDER, ordered=True)
    return out

dfs = prep(dfseqid)
dfr = prep(dfretention)

# ── Figure ───────────────────────────────────────────────────────────────────
# Nature double-column width: 183 mm ≈ 7.20 in
fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.6),
                         gridspec_kw={"wspace": 0.42})

configs = [
    (dfs, "Sequence identity",         0.0, 0.95),
    (dfr, "Functional site retention", 0.0, 1.18),
]
colors = [PALETTE[m] for m in MODEL_ORDER]

for ax, (df, ylabel, ymin, ymax) in zip(axes, configs):
    sns.barplot(
        data=df, x="protein", y="score",
        hue="model", hue_order=MODEL_ORDER, order=PROT_ORDER,
        palette=colors, ax=ax, width=0.65,
        linewidth=0, saturation=1.0,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    ax.get_legend().remove()
    ax.tick_params(axis="x", bottom=False, pad=2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.grid(True, linewidth=0.3, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

for ax, lbl in zip(axes, "ab"):
    ax.text(-0.20, 1.06, lbl, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")

handles = [mpatches.Patch(facecolor=PALETTE[m], label=m, linewidth=0)
           for m in MODEL_ORDER]
fig.legend(handles=handles, ncol=4, frameon=False,
           loc="upper center", bbox_to_anchor=(0.5, 1.10),
           handlelength=1.0, handleheight=0.85, columnspacing=0.8)

fig.savefig("comparison_nature.svg", bbox_inches="tight")
fig.savefig("comparison_nature.png", bbox_inches="tight", dpi=300)
print("Saved → comparison_nature.svg / comparison_nature.png")
