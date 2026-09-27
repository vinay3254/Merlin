#!/usr/bin/env python3
"""Generate architecture and benchmark progression images for README."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor("#0f172a") # dark slate
    ax.set_facecolor("#0f172a")

    # Pipeline stages
    stages = [
        {
            "title": "1. Multi-Source Ingestion",
            "desc": "• Source 1: Deduplicated Ref (1.73M)\n• Source 2: Noisy Secondary (5.0M)\n• Source 3: Noisy Secondary (4.9M)\n• Unseen Test Domains (France)",
            "color": "#38bdf8", # sky blue
            "box_color": "#1e293b",
            "x": 0.5, "y": 5.0, "w": 3.8, "h": 2.2
        },
        {
            "title": "2. Preprocessing & Skeletons",
            "desc": "• Accent folding (NFKD)\n• Indic Unicode retention\n• Legal form core separation\n• Phonetic consonant skeletons",
            "color": "#818cf8", # indigo
            "box_color": "#1e293b",
            "x": 4.8, "y": 5.0, "w": 3.8, "h": 2.2
        },
        {
            "title": "3. Sparse Multi-Blocking",
            "desc": "• Hashed-token CSR sparse matrix\n• Country-partitioned cosine top-K\n• Combined + Name + Address union\n• 96.34% Ground Truth Recall",
            "color": "#a78bfa", # purple
            "box_color": "#1e293b",
            "x": 9.1, "y": 5.0, "w": 3.8, "h": 2.2
        },
        {
            "title": "4. Feature Engineering",
            "desc": "• RapidFuzz cpdist SIMD metrics\n• Token & postal overlap counts\n• Cross-entity competitive ranking\n• Column-major disk memmaps",
            "color": "#f472b6", # pink
            "box_color": "#1e293b",
            "x": 9.1, "y": 1.5, "w": 3.8, "h": 2.2
        },
        {
            "title": "5. Hybrid Classifiers",
            "desc": "• Stage 1: LightGBM GBDT (200k)\n• Stage 2: Multilingual MiniLM-L12\n  Cross-Encoder on hard pairs\n• Logistic logit blending",
            "color": "#fb923c", # orange
            "box_color": "#1e293b",
            "x": 4.8, "y": 1.5, "w": 3.8, "h": 2.2
        },
        {
            "title": "6. Decoding & Optimization",
            "desc": "• Sibling-consistency re-scoring\n• Best-S1 record assignment\n• Macro F_0.5 decision threshold\n• Format & ID schema validation",
            "color": "#4ade80", # emerald green
            "box_color": "#1e293b",
            "x": 0.5, "y": 1.5, "w": 3.8, "h": 2.2
        },
    ]

    # Draw boxes
    for s in stages:
        box = patches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle="round,pad=0.2,rounding_size=0.15",
            edgecolor=s["color"],
            facecolor=s["box_color"],
            linewidth=2.0,
            zorder=2
        )
        ax.add_patch(box)
        
        # Title
        ax.text(
            s["x"] + s["w"]/2, s["y"] + s["h"] - 0.35,
            s["title"],
            ha="center", va="center",
            color=s["color"],
            fontsize=12, fontweight="bold",
            zorder=3
        )
        
        # Details
        ax.text(
            s["x"] + 0.25, s["y"] + s["h"]/2 - 0.2,
            s["desc"],
            ha="left", va="center",
            color="#e2e8f0",
            fontsize=9.5, linespacing=1.4,
            zorder=3
        )

    # Connecting arrows
    arrow_props = dict(arrowstyle="-|>", color="#94a3b8", lw=2.5, mutation_scale=18)
    
    # 1 -> 2
    ax.annotate("", xy=(4.75, 6.1), xytext=(4.35, 6.1), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(9.05, 6.1), xytext=(8.65, 6.1), arrowprops=arrow_props)
    # 3 -> 4 (down)
    ax.annotate("", xy=(11.0, 3.75), xytext=(11.0, 4.95), arrowprops=arrow_props)
    # 4 -> 5 (left)
    ax.annotate("", xy=(8.65, 2.6), xytext=(9.05, 2.6), arrowprops=arrow_props)
    # 5 -> 6 (left)
    ax.annotate("", xy=(4.35, 2.6), xytext=(4.75, 2.6), arrowprops=arrow_props)

    # Title
    ax.text(
        6.7, 8.1,
        "Kinfolk: Scalable Multi-Source Business Entity Resolution Architecture",
        ha="center", va="center",
        color="#f8fafc",
        fontsize=16, fontweight="bold"
    )
    ax.text(
        6.7, 7.6,
        "End-to-End Sparse Blocking, SIMD Features, GBDT + Multilingual Cross-Encoder Pipeline",
        ha="center", va="center",
        color="#94a3b8",
        fontsize=11
    )

    ax.set_xlim(-0.2, 13.6)
    ax.set_ylim(0.5, 8.6)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("docs/images/architecture.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved docs/images/architecture.png")


def generate_benchmark_chart():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    versions = [
        "v2 Baseline\n(Inverted Index)",
        "v2 + Rule\n(1-S1/Record)",
        "v3 Sparse\n(CSR Blocking)",
        "v3b Phonetic\n(Skeletons)",
        "v3b + CE\n(MiniLM)",
        "v5 + CE + S2\n(Sibling Consistency)",
        "v7 Final\n(French Pseudo-Labels)"
    ]
    
    scores = [0.789, 0.802, 0.930, 0.934, 0.943, 0.953, 0.978]
    colors = ["#64748b", "#64748b", "#38bdf8", "#818cf8", "#a855f7", "#22c55e", "#eab308"]

    x = np.arange(len(versions))
    bars = ax.bar(x, scores, color=colors, width=0.55, edgecolor="#334155", linewidth=1.5, zorder=3)

    # Line plot overlay
    ax.plot(x, scores, color="#f8fafc", linestyle="--", linewidth=2, marker="o", markersize=8, zorder=4, alpha=0.85)

    # Value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        label = f"{score:.3f}"
        if i == len(scores) - 1:
            label += "\n(Val)"
        ax.text(
            bar.get_x() + bar.get_width()/2,
            score + 0.015,
            label,
            ha="center", va="bottom",
            color="#f8fafc",
            fontsize=10.5, fontweight="bold",
            zorder=5
        )

    ax.set_ylim(0.70, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=9.5, color="#cbd5e1")
    ax.tick_params(axis="y", colors="#cbd5e1", labelsize=10)
    ax.grid(axis="y", color="#334155", linestyle=":", alpha=0.7, zorder=1)

    # Labels and Title
    ax.set_ylabel("Macro-averaged F0.5 Score", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title("Leaderboard & Validation Progression (Amazon ML Challenge 2026)", color="#f8fafc", fontsize=14, fontweight="bold", pad=15)

    # Spines styling
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig("docs/images/leaderboard_progress.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved docs/images/leaderboard_progress.png")

if __name__ == "__main__":
    generate_architecture_diagram()
    generate_benchmark_chart()
