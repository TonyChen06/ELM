"""Paper-style figures for the tonytry README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# palette
INK = "#1f2430"
BLUE, BLUE_D = "#dbeafe", "#1d4ed8"     # data
GREEN, GREEN_D = "#dcfce7", "#15803d"   # encoders/model
ORANGE, ORANGE_D = "#ffedd5", "#c2410c" # llm
PURPLE, PURPLE_D = "#ede9fe", "#6d28d9" # heads
GRAY, GRAY_D = "#f1f5f9", "#475569"


def box(ax, x, y, w, h, title, lines=(), fc=GRAY, ec=GRAY_D, title_size=11, body_size=9, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
                                fc=fc, ec=ec, lw=1.6, linestyle="--" if dashed else "-"))
    cy = y + h - 0.34 if lines else y + h / 2
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=INK)
    if lines:
        body = "\n".join(lines)
        ax.text(x + w / 2, y + (h - 0.55) / 2, body, ha="center", va="center",
                fontsize=body_size, color=GRAY_D, linespacing=1.45)
    return (x, y, w, h)


def arrow(ax, p1, p2, color=GRAY_D, lw=2.0, dashed=False, label=None, label_dy=0.16, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, linestyle=(0, (5, 4)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.5, color=color, style="italic")


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


# ============================ Figure 1: architecture ========================
fig, ax = plt.subplots(figsize=(17.5, 8.6))
ax.set_xlim(0, 17.5)
ax.set_ylim(0, 8.6)
ax.axis("off")

ecg = box(ax, 0.35, 5.3, 2.1, 1.6, "12-lead ECG", ["float32 [12 × 2500]", "noise / flatline / perturb"], fc=BLUE, ec=BLUE_D)
conv = box(ax, 0.35, 1.5, 2.1, 1.6, "Conversation", ["HF datasets, mixed lazily", "multi-turn QA / reports"], fc=BLUE, ec=BLUE_D)

rep_y, rep_h = 4.55, 0.62
reps = {}
for i, (name, note) in enumerate([("signal", "raw matrix"),
                                  ("stacked_signal", "3-channel image"),
                                  ("rgb", "clinical plot"),
                                  ("symbolic", "ECG-Byte BPE")]):
    reps[name] = box(ax, 3.1, rep_y + i * 0.78, 2.35, rep_h, name, (), fc=BLUE, ec=BLUE_D, title_size=9.5)
ax.text(4.27, 7.62, "representations (strategy objects)", ha="center", fontsize=9, color=BLUE_D, fontweight="bold")

enc = box(ax, 6.25, 4.9, 2.7, 2.5, "Encoder  (optional)",
          ["vit1d: st_mem · mtae · mlae", "merl (ResNet-101 1-D)", "CLIP · SigLIP2 · ViT"], fc=GREEN, ec=GREEN_D)
con = box(ax, 9.75, 5.35, 2.1, 1.6, "Connector",
          ["linear · mlp", "patch · cnn-patch"], fc=GREEN, ec=GREEN_D)

tmpl = box(ax, 3.1, 1.5, 2.7, 1.6, "templates.py",
           ["tokenizer's own chat template", "prefix-diff label masking"], fc=BLUE, ec=BLUE_D)
emb = box(ax, 6.6, 1.5, 2.35, 1.6, "Token embeddings", ["unpadded items;", "collator pads to batch max"], fc=ORANGE, ec=ORANGE_D)
inj = box(ax, 9.75, 1.95, 2.1, 1.5, "Inject", ["connector output at", "<signal> positions"], fc=ORANGE, ec=ORANGE_D)
llm = box(ax, 12.7, 2.6, 2.3, 2.9, "HF causal LM", ["Llama 3.2 · Qwen 2.5", "Gemma 2  (any chat LLM)", "LoRA · compile · SDPA"], fc=ORANGE, ec=ORANGE_D)

heads_x = 15.7
sft = box(ax, heads_x, 5.6, 1.6, 1.0, "SFT / pretrain", ["loss step"], fc=PURPLE, ec=PURPLE_D, title_size=9.5, body_size=8.5)
rl = box(ax, heads_x, 4.1, 1.6, 1.0, "RL (SAPO)", ["group rollouts"], fc=PURPLE, ec=PURPLE_D, title_size=9.5, body_size=8.5)
ev = box(ax, heads_x, 2.6, 1.6, 1.0, "Eval / chat", ["batched · N-turn"], fc=PURPLE, ec=PURPLE_D, title_size=9.5, body_size=8.5)

mesh = box(ax, 5.6, 0.25, 9.9, 0.75, "dist.py — DDP | FSDP2 (DeviceMesh, mixed precision), collective checkpointing, tokens/s + MFU",
           (), fc=GRAY, ec=GRAY_D, title_size=9.5, dashed=True)

# arrows: ECG -> representations
for name in reps:
    arrow(ax, right(ecg), left(reps[name]), color=BLUE_D, rad=0.06)
# representations -> encoder / direct paths
arrow(ax, right(reps["stacked_signal"]), (6.25, 5.8), color=GREEN_D, rad=-0.05)
arrow(ax, right(reps["rgb"]), (6.25, 6.5), color=GREEN_D, rad=-0.05)
arrow(ax, right(reps["signal"]), (6.25, 5.2), color=GREEN_D, rad=0.0)
arrow(ax, right(enc), left(con), color=GREEN_D)
ax.plot([5.45, 9.35], [rep_y + 0.18, 4.42], color=GREEN_D, lw=2.0, linestyle=(0, (5, 4)), zorder=1)
arrow(ax, (9.35, 4.42), (10.62, 5.32), color=GREEN_D, dashed=True)
ax.text(7.35, 4.18, "ELF: raw signal, no encoder", fontsize=8.5, color=GREEN_D, style="italic", ha="center")
arrow(ax, (3.4, rep_y + 3 * 0.78), (4.6, 3.1), color=BLUE_D, dashed=True, rad=0.35)
ax.text(2.95, 3.95, "ecg_byte: BPE ids\nspliced into prompt", fontsize=8.5, color=BLUE_D, style="italic", ha="center")
# text path
arrow(ax, right(conv), left(tmpl), color=BLUE_D)
arrow(ax, right(tmpl), left(emb), color=BLUE_D)
arrow(ax, right(emb), left(inj), color=ORANGE_D)
arrow(ax, bottom(con), top(inj), color=GREEN_D)
arrow(ax, right(inj), (12.7, 3.6), color=ORANGE_D)
# llm -> heads
arrow(ax, right(llm), left(sft), color=PURPLE_D, rad=-0.12)
arrow(ax, right(llm), left(rl), color=PURPLE_D)
arrow(ax, right(llm), left(ev), color=PURPLE_D, rad=0.12)

ax.text(0.35, 8.38, "ELM architecture (tonytry)", fontsize=15, fontweight="bold", color=INK)
ax.text(0.35, 8.04, "one composed model: [encoder] → [connector] → LM; components declared in registry.py",
        fontsize=10, color=GRAY_D)
fig.savefig("assets/architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ===================== Figure 2: lines per subsystem ========================
subsystems = [
    ("templating & masking", 590, 158, "vendored FastChat + watch-token tables → tokenizer's own template"),
    ("data pipeline", 925, 340, "4 copy-pasted dataset classes → 1 dataset + strategy objects; padding in one place"),
    ("ECG / vision encoders", 1088, 166, "st_mem + mtae + mlae → one masked ViT; 3 vision wrappers → 1"),
    ("model glue", 855, 175, "3 identical LLM wrappers + 2 injection modules + 3 builders → Elm + registry"),
    ("config & registries", 480, 279, "argparse sprawl + mutable global dicts → one dataclass + registry"),
    ("optimizers & trainer", 370, 222, "2 near-identical loops → 1 loop, RL as a pluggable step"),
    ("RL (SAPO)", 395, 123, "vectorized rollout masking; loss summed once per batch"),
    ("evaluation", 470, 305, "turn-flattened, batched, multi-GPU (was bs=1 per turn)"),
    ("mains · chat · dist", 750, 239, "chat reuses templates/model; DDP+FSDP2 seam replaces utils sprawl"),
]
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ys = range(len(subsystems))[::-1]
old_vals = [s[1] for s in subsystems]
new_vals = [s[2] for s in subsystems]
bh = 0.38
ax.barh([y + bh / 2 + 0.02 for y in ys], old_vals, height=bh, color="#cbd5e1", label="before (main): 5,923 lines")
ax.barh([y - bh / 2 - 0.02 for y in ys], new_vals, height=bh, color="#2563eb", label="after (tonytry): 2,007 lines")
for y, (name, old, new, note) in zip(ys, subsystems):
    ax.text(-18, y, name, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(old + 14, y + bh / 2 + 0.02, f"{old:,}", va="center", fontsize=9, color=GRAY_D)
    ax.text(new + 14, y - bh / 2 - 0.02, f"{new:,}", va="center", fontsize=9, color="#1d4ed8", fontweight="bold")
    ax.text(1130, y, note, va="center", fontsize=8.6, color=GRAY_D, style="italic")
ax.set_xlim(0, 2600)
ax.set_ylim(-0.6, len(subsystems) - 0.2)
ax.axis("off")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=10.5)
ax.set_title("Lines of code per subsystem — every feature kept, plus FSDP2 / collective ckpt / MFU / compile",
             fontsize=12.5, color=INK, loc="left", pad=14)
fig.savefig("assets/lines.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote assets/architecture.png and assets/lines.png")
