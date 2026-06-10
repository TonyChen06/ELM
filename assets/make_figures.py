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

# ================= Figure 0: BEFORE architecture (main) =====================
RED, RED_D = "#fee2e2", "#b91c1c"

fig, ax = plt.subplots(figsize=(17.5, 9.8))
ax.set_xlim(0, 17.5)
ax.set_ylim(0, 9.8)
ax.axis("off")

ecg_b = box(ax, 0.35, 5.6, 1.95, 1.5, "12-lead ECG", ["float64 [12 × 2500]"], fc=BLUE, ec=BLUE_D, body_size=8.5)
conv_b = box(ax, 0.35, 1.7, 1.95, 1.5, "Conversation", ["HF datasets,", "materialized to RAM"], fc=BLUE, ec=BLUE_D, body_size=8.5)

# 4 dataset classes, each carrying its own copy of the prep methods
ax.text(4.0, 8.5, "4 dataset classes — prepare/truncate/pad COPY-PASTED in each", ha="center",
        fontsize=9, color=RED_D, fontweight="bold")
ds_boxes = []
for i, name in enumerate(["symbolic.py", "rgb.py", "stacked_signal.py", "signal.py"]):
    ds_boxes.append(box(ax, 2.85, 4.7 + i * 0.95, 2.3, 0.78, name,
                        ["prepare · trunc · pad"], fc=RED, ec=RED_D, title_size=8.5, body_size=7.5))
pad = box(ax, 2.85, 3.6, 2.3, 0.7, "pad every item to 2048", (), fc=GRAY, ec=GRAY_D, title_size=8.5, dashed=True)

fastchat = box(ax, 2.85, 1.45, 2.6, 1.7, "chat_template_manager",
               ["428 lines of vendored FastChat", "(3 of ~20 templates used)"], fc=RED, ec=RED_D,
               title_size=9.5, body_size=8)

constants = box(ax, 6.1, 7.45, 3.3, 1.25, "configs/constants.py  (385)",
                ["HF_LLMS · ENCODERS dicts, mutated by builders;", "watch-token id tables read by every dataset class"],
                fc=RED, ec=RED_D, title_size=9.5, body_size=7.8)

builders = box(ax, 6.1, 5.45, 3.3, 1.55, "builders  (406)",
               ["build_elm → build_llm / build_encoder", "→ connect_nns   (if/elif chains ×3)"],
               fc=GRAY, ec=GRAY_D, title_size=9.5, body_size=8)

# encoder cluster with cross-imports
ax.text(7.5, 4.95, "encoders share blocks by cross-import", ha="center", fontsize=8.5, color=RED_D, style="italic")
st = box(ax, 6.1, 3.85, 1.45, 0.75, "st_mem", ["422"], fc=GREEN, ec=GREEN_D, title_size=8.5, body_size=7.5)
mt = box(ax, 7.75, 3.85, 1.45, 0.75, "mtae", ["224"], fc=GREEN, ec=GREEN_D, title_size=8.5, body_size=7.5)
ml = box(ax, 6.9, 2.85, 1.45, 0.75, "mlae", ["197"], fc=GREEN, ec=GREEN_D, title_size=8.5, body_size=7.5)
merl_b = box(ax, 9.45, 3.85, 1.35, 0.75, "merl", ["169 (+dead)"], fc=GREEN, ec=GREEN_D, title_size=8.5, body_size=7.5)
vis = box(ax, 9.45, 2.85, 1.35, 0.75, "vision ×3", ["near-identical"], fc=RED, ec=RED_D, title_size=8.5, body_size=7.5)
arrow(ax, right(st), left(mt), color=GREEN_D, dashed=True)
arrow(ax, bottom(st), top(ml), color=GREEN_D, dashed=True)
arrow(ax, bottom(mt), top(ml), color=GREEN_D, dashed=True)

# 3 identical wrappers
ax.text(11.4, 8.42, "IDENTICAL ×3", ha="center", fontsize=9, color=RED_D, fontweight="bold")
for i, name in enumerate(["gemma2.py", "qwen25.py", "llama3.py"]):
    box(ax, 10.5, 6.5 + i * 0.62, 1.8, 0.5, name, (), fc=RED, ec=RED_D, title_size=8.5)

# 2 injection modules
ax.text(11.4, 5.95, "injection COPY-PASTED ×2", ha="center", fontsize=9, color=RED_D, fontweight="bold")
llava_b = box(ax, 10.5, 5.05, 1.8, 0.62, "llava.py  95", (), fc=RED, ec=RED_D, title_size=8.5)
elf_b = box(ax, 10.5, 4.3, 1.8, 0.62, "base_elf.py  89", (), fc=RED, ec=RED_D, title_size=8.5)

llm_b = box(ax, 13.0, 5.2, 2.1, 2.2, "HF causal LM", ["bf16, sdpa", "(encoders: manual", "attention, no SDPA)"],
            fc=ORANGE, ec=ORANGE_D, title_size=10.5, body_size=8.5)

# heads: duplicated loops, slow eval
tr1 = box(ax, 15.6, 6.6, 1.7, 0.85, "trainer.py", ["75"], fc=PURPLE, ec=PURPLE_D, title_size=9, body_size=7.5)
tr2 = box(ax, 15.6, 5.55, 1.7, 0.85, "rl_trainer.py", ["77 — ≈same loop"], fc=RED, ec=RED_D, title_size=9, body_size=7.5)
ev_b = box(ax, 15.6, 4.1, 1.7, 1.1, "evaluator.py", ["392 — bs=1,", "1 generate / turn"], fc=PURPLE, ec=PURPLE_D, title_size=9, body_size=7.5)
chat_b = box(ax, 15.6, 2.7, 1.7, 1.1, "main_chat.py", ["233 — re-implements", "tokenizer/EOS/decode"], fc=RED, ec=RED_D, title_size=9, body_size=7.5)

utils = box(ax, 2.85, 0.3, 12.0, 0.75,
            "utils/ ≈600 — gpu · checkpoint · dir · seed · time · viz · wandb managers  (+ dead code: timeit, log_wandb, plot_signals, AttentionPool2d, …)",
            (), fc=GRAY, ec=GRAY_D, title_size=9, dashed=True)

# flows
for b in ds_boxes:
    arrow(ax, right(ecg_b), left(b), color=BLUE_D, rad=0.06)
arrow(ax, right(conv_b), left(fastchat), color=BLUE_D)
ax.plot([5.45, 13.4], [2.15, 2.35], color=BLUE_D, lw=2.0, zorder=1)
arrow(ax, (13.4, 2.35), (14.05, 5.2), color=BLUE_D)
arrow(ax, right(pad), (6.1, 4.15), color=GREEN_D, rad=0.05)
arrow(ax, right(builders), (10.5, 6.0), color=GRAY_D, rad=0.1)
arrow(ax, (9.4, 6.2), (10.5, 7.0), color=GRAY_D, rad=-0.1)
arrow(ax, (12.3, 7.0), (13.5, 6.6), color=ORANGE_D, rad=-0.1)
arrow(ax, right(llava_b), (13.0, 5.6), color=ORANGE_D)
arrow(ax, right(llm_b), left(tr1), color=PURPLE_D, rad=-0.1)
arrow(ax, right(llm_b), left(tr2), color=PURPLE_D)
arrow(ax, right(llm_b), left(ev_b), color=PURPLE_D, rad=0.08)
arrow(ax, right(llm_b), left(chat_b), color=RED_D, rad=0.15)

# mutable-global side channels (red dashed)
arrow(ax, top(builders), bottom(constants), color=RED_D, dashed=True)
ax.text(7.95, 7.16, "writes model_hidden_size, projection_dim", fontsize=7.5,
        color=RED_D, style="italic", ha="left")
arrow(ax, (9.4, 7.85), (10.5, 7.45), color=RED_D, dashed=True, rad=-0.12)
arrow(ax, (6.35, 7.45), (5.0, 7.0), color=RED_D, dashed=True, rad=0.15)

ax.text(13.2, 1.45, '"needs signal injection?" → hand-written ELM lists in evaluator,\nmain_chat ×2 — all three disagree (2 of them crash)',
        fontsize=8.5, color=RED_D, ha="center", style="italic")

ax.text(0.35, 9.55, "ELM architecture (before, `main`)", fontsize=15, fontweight="bold", color=INK)
ax.text(0.35, 9.22, "red = duplicated code or mutable global state;  dashed red = runtime side-channels through constants.py",
        fontsize=10, color=GRAY_D)
fig.savefig("assets/architecture_before.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote assets/architecture_before.png")
