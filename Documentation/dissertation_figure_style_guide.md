# Dissertation Figure Style Guide
*Inspired by DeepMind / Google DeepMind visual language*

---

## 1. Design Philosophy

DeepMind figures share a distinctive visual language across their landmark papers (DQN, AlphaGo, AlphaFold): they are **information-dense but never cluttered**, **restrained in colour but never dull**, and always **subordinated to the data**. Every pixel earns its place.

Three core principles underpin this guide:

**Clarity over decoration.** Gridlines are minimal or absent. Chartjunk (unnecessary 3D effects, drop shadows, heavy borders) is eliminated entirely. The figure should be readable at 60% zoom.

**Consistent visual identity.** A single colour palette, typeface family, and line-weight system runs across every figure in the dissertation. The reader should feel the work was designed as a whole.

**Figures are arguments.** Every figure has a clear thesis — a single main takeaway that the design amplifies. Panel arrangement, annotation, and colour all serve this argument.

---

## 2. Colour Palette

### 2.1 Primary Palette

DeepMind figures use a small, controlled palette with a signature **teal/cyan** as the primary data colour, complemented by muted secondary colours. Use this exact palette for consistency.

| Role | Name | Hex | RGB | When to Use |
|---|---|---|---|---|
| Primary data | DeepMind Teal | `#00BCD4` | 0, 188, 212 | Main model / your method |
| Secondary data | Slate Blue | `#5C6BC0` | 92, 107, 192 | Baselines, comparisons |
| Tertiary data | Coral | `#EF5350` | 239, 83, 80 | Ablations, negative results |
| Tertiary alt | Amber | `#FFA726` | 255, 167, 38 | Third comparison line |
| Success / high | Forest Green | `#43A047` | 67, 160, 71 | High-confidence, "correct" |
| Neutral fill | Light Grey | `#ECEFF1` | 236, 239, 241 | Background fills, shading |
| Annotation | Dark Charcoal | `#37474F` | 55, 71, 79 | Labels, annotation text |
| Background | Off-White | `#FAFAFA` | 250, 250, 250 | Figure background |

### 2.2 Extended Palette for Multi-Category Plots

When more than four categories are needed, extend the palette in this order. **Never use more than 7 distinct hues in a single figure.**

```
1. #00BCD4  (Teal)
2. #5C6BC0  (Slate Blue)
3. #EF5350  (Coral)
4. #FFA726  (Amber)
5. #43A047  (Forest Green)
6. #AB47BC  (Violet)
7. #78909C  (Blue-Grey)
```

### 2.3 Sequential / Diverging Scales

For heatmaps, value functions, and confidence maps — as seen in the DQN t-SNE figure and AlphaFold pLDDT colouring:

**Sequential (low → high):** Dark blue → Teal → Yellow-White
```
#1A237E → #0288D1 → #00BCD4 → #80DEEA → #FFFFFF
```

**Diverging (negative → zero → positive):** Deep Blue → White → Deep Red
```
#1565C0 → #90CAF9 → #FFFFFF → #EF9A9A → #B71C1C
```

**Confidence / pLDDT-style (low → high):**
```
#D50000 → #FF6D00 → #FFD600 → #00C853  (Red → Orange → Yellow → Green)
```

### 2.4 Colour Accessibility

- All colour pairs used for data discrimination must pass **WCAG AA contrast** against both white and `#FAFAFA` backgrounds.
- Never encode information in colour alone. Supplement with line style, marker shape, or direct labelling.
- Check figures in greyscale before finalising. The ordering of lines/bars must still be distinguishable.

---

## 3. Typography

### 3.1 Font Stack

| Context | Font | Weight | Size |
|---|---|---|---|
| Figure title (bold label e.g. "Fig. 3") | **Inter** (or Helvetica Neue) | 600 Semi-bold | 9 pt |
| Axis labels | Inter | 400 Regular | 8 pt |
| Tick labels | Inter | 400 Regular | 7 pt |
| Legend text | Inter | 400 Regular | 7.5 pt |
| In-figure annotation | Inter | 400 Regular | 7–8 pt |
| Equation in figure | Latin Modern Math / Computer Modern | Regular | 8 pt |
| Caption body (in main text) | Matches dissertation body font | Regular | 9–10 pt |

> **Fallback stack:** Inter → Helvetica Neue → Arial → sans-serif

### 3.2 Typography Rules

- **No serif fonts** inside figures. Serif fonts belong in the dissertation body text, not axes or labels.
- **No all-caps** axis labels. Use sentence case: "Training epochs" not "TRAINING EPOCHS".
- **Axis label orientation:** x-axis labels are horizontal; y-axis labels are rotated 90° counter-clockwise. Never rotate tick labels more than 45° — if they overlap, abbreviate or use a categorical layout.
- **Figure labels (a, b, c...)** use **bold lower-case** in parentheses: **(a)**, **(b)**. Placed in the top-left corner of each panel. As seen in AlphaFold Fig. 1 and AlphaGo Fig. 4.
- **Numbers on axes** always use the same decimal precision within a panel.

---

## 4. Line Weights and Marker Styles

### 4.1 Lines

| Element | Weight |
|---|---|
| Primary data line | 2.0 pt |
| Secondary / baseline lines | 1.5 pt |
| Gridlines (major) | 0.5 pt, `#CFD8DC`, dashed `[4, 4]` |
| Gridlines (minor) | Omit entirely |
| Axis spines (bottom + left only) | 1.0 pt, `#37474F` |
| Axis spines (top + right) | **Remove entirely** |
| Error bar / confidence interval | 1.0 pt, same colour as parent line |
| Annotation arrow | 1.0 pt, `#37474F`, arrowhead size 4 pt |

> **Remove top and right spines** on all line/scatter/bar plots. This is the single most impactful change for a clean, modern look — it is consistent across every DeepMind figure.

### 4.2 Markers

Use markers sparingly — only when individual data points matter (e.g. ablation tables plotted as scatter).

| Marker | Use Case |
|---|---|
| `○` Circle (filled) | Primary method |
| `□` Square (filled) | Baseline 1 |
| `△` Triangle-up (filled) | Baseline 2 |
| `◇` Diamond (filled) | Ablation |
| `×` Cross | Human / oracle reference |

Marker size: **6–8 pt** for scatter; **omit on dense line plots**.

---

## 5. Layout and Spacing

### 5.1 Figure Dimensions

| Figure Type | Width | Height |
|---|---|---|
| Single-column figure | 88 mm (3.46 in) | As needed, ≤ 100 mm |
| Full-width figure | 180 mm (7.09 in) | As needed, ≤ 120 mm |
| Multi-panel (2-col) | 180 mm total, panels equal | As needed |
| Multi-panel (3-col) | 180 mm total | As needed |

Export at **300 DPI minimum** for print; **600 DPI** for figures with fine detail (architecture diagrams, attention maps).

### 5.2 White Space and Margins

- **Axis padding:** Leave at least 5% whitespace between the outermost data point and the plot edge.
- **Panel spacing:** In multi-panel figures, use consistent inter-panel gaps of **8–12 mm**.
- **Legend placement:** Inside the plot area where space allows (top-right or bottom-right, away from data). External legends only when the plot is too dense. No legend box border.
- **Caption separation:** Figures should sit with the caption directly beneath, separated by 4 pt of vertical space.

### 5.3 Panel Label Placement

Following the AlphaFold/AlphaGo convention:

```
(a) — bold, 9pt, positioned 2mm left and 2mm above the plot area,
       or at top-left corner of the panel with 2mm internal padding.
```

---

## 6. Specific Figure Types

### 6.1 Training / Learning Curves

*Reference: DQN Fig. 2 — training curves for Space Invaders and Seaquest.*

- Use a **thin light-grey band** (alpha = 0.15) for standard deviation or confidence interval. Do not use error bars on dense time-series.
- x-axis: "Training steps (×10⁶)" or "Epochs". Always state units.
- y-axis: "Average score per episode" or the precise metric. State normalisation if used.
- Include a **horizontal dashed reference line** in `#9E9E9E` for baselines (random policy, human performance), with a direct text label on the line rather than a legend entry.
- If plotting multiple games/tasks on a grid, keep axis ranges identical across panels.

```python
# matplotlib template for training curves
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial'],
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 2.0,
    'legend.frameon': False,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

TEAL   = '#00BCD4'
SLATE  = '#5C6BC0'
CORAL  = '#EF5350'
GREY   = '#78909C'
```

### 6.2 Bar Charts / Comparisons

*Reference: AlphaGo Fig. 4a — Elo rating bar chart.*

- **Horizontal bars** preferred when comparing many named methods (avoids label rotation).
- **Vertical bars** for temporal or ordinal categories.
- Bar fill: use the primary palette. Your method gets the primary colour; baselines get muted shades.
- **Highlight your method** with a slightly darker border (`linewidth=1.0`) or a star/annotation, rather than a different colour.
- Error bars: cap style `|`, linewidth 1.0 pt, colour matches bar.
- No 3D effects, gradients, or patterns.

### 6.3 Scatter Plots and Correlation Plots

*Reference: AlphaFold Fig. 2c–d — pLDDT vs lDDT-Cα, pTM vs TM-score.*

- Include a **diagonal reference line** (y = x, or least-squares fit) in `#9E9E9E`, linewidth 1.0 pt, linestyle `--`.
- Add **Pearson's r** and **n** as annotation in the top-left or bottom-right of the plot, 7pt, charcoal.
- Use **hexbin** or **kernel density contours** when n > 500 to avoid overplotting. Colour scale: white → teal (sequential).
- Zoom inset panels (as in AlphaFold Fig. 2c,d) should have a thin box border in `#90A4AE`.

### 6.4 Architecture and Pipeline Diagrams

*Reference: AlphaGo Fig. 1, AlphaFold Fig. 1e and Fig. 3.*

- Use a **consistent box style**: rounded rectangles (`border-radius: 4px`), `#ECEFF1` fill, `#90A4AE` border, 1 pt.
- **Module boxes** for main components: white fill, `#37474F` border, 1.5 pt, bold label inside.
- **Data flow arrows**: `#5C6BC0`, linewidth 1.5 pt, filled arrowhead. Bidirectional arrows for information that flows in both directions.
- **Colour-code functional groups**: use soft background rectangles (alpha = 0.08) to group related modules — training pipeline (teal tint), inference path (slate tint), data sources (amber tint).
- Array shape annotations: grey italic text (`#78909C`, 7 pt) adjacent to arrows, in parentheses: e.g. `(N_res, N_res, c)`.
- Use consistent left-to-right or top-to-bottom flow. Never mix flow directions within a single panel.
- Software tools: **draw.io**, Inkscape, or TikZ/PGF in LaTeX. Export as **PDF vector** for LaTeX; **SVG → PDF** for maximum sharpness.

### 6.5 Heatmaps and Attention Maps

*Reference: DQN Fig. 4 (t-SNE), AlphaFold attention maps.*

- Use the **sequential or diverging palettes** from §2.3. Never use rainbow/jet colourmaps.
- Include a **colourbar** with concise label. Position: right side, same height as the heatmap.
- Colourbar ticks: 3–5 ticks maximum.
- For attention maps overlaid on sequences/structures: use a **semi-transparent overlay** (alpha = 0.6) so the underlying structure remains visible.

### 6.6 Ablation Tables Plotted as Figures

*Reference: AlphaFold Fig. 4a — ablation results with dot-and-whisker.*

- Use a **horizontal dot plot** (lollipop chart): horizontal line from zero to the point, filled circle at the value.
- Positive changes (improvements): teal. Negative changes: coral.
- Baseline row: separate horizontal dashed line at x = 0, labelled "Baseline".
- Row labels on the y-axis, right-aligned in the plot frame.
- This is more compact and readable than grouped bar charts for ablations.

### 6.7 Protein / 3D Structure Figures

*Reference: AlphaFold Figs. 1b–d, 5b.*

- Use **PyMOL** or **ChimeraX** for structure rendering.
- Colour scheme: predicted structure in **teal** (`#00BCD4`); experimental/true structure in **light green** (`#66BB6A`).
- Background: **white** or **very light grey** (`#F5F5F5`). Never black or dark backgrounds in dissertation figures.
- Remove default PyMOL background gradient. Use `bg_color white`.
- Secondary structure colouring (if not showing comparison): alpha helices in teal, beta sheets in slate blue, loops in light grey.
- Include a **scale bar** (in Å or nm) rather than relying on the reader to infer scale.
- State the PDB accession code in the caption, not on the figure itself.

---

## 7. Equations and Mathematical Notation in Figures

- Mathematical notation within figure panels (axis labels, annotations) should use **LaTeX-rendered text** where possible (`matplotlib`'s `mathtext` or `usetex=True`).
- Use the same notation as in the main text. Never introduce new symbols in a figure without defining them.
- Fractions in axis labels should be written inline: `Loss / N` not `Loss/N`.

---

## 8. Captions

Caption style follows the DeepMind convention closely:

```
Fig. 3 | Descriptive title in bold sentence case.
Panel-by-panel description using (a), (b), (c) references.
Statistical details (n, confidence intervals, test used) stated
explicitly. Data are [mean / median] ± [s.d. / s.e.m. / 95% CI].
n = [number]. All error bars represent [what].
```

Key rules:
- **Bold "Fig. N |"** followed by a pipe character, then the title.
- Panels are described **in order**: "(a) ..., (b) ..., (c) ...".
- State exactly what error bars/bands represent. This is non-negotiable.
- Cite any data sources, external tools, or permissions (e.g. "With permission from X").
- Captions live **below** the figure, not above.

---

## 9. Referencing Human / Oracle Performance

Following the DQN convention (100% human, 0% random normalisation):

- Plot a **horizontal dashed line** at the human/oracle level.
- Label it directly on the line: "Human" or "Oracle" in 7pt, matching the line colour.
- If normalising performance as a percentage: state clearly in the caption and y-axis label what 100% and 0% correspond to.

---

## 10. Export and File Management

### 10.1 File Formats

| Use Case | Format | Notes |
|---|---|---|
| LaTeX dissertation | PDF (vector) | Via `savefig('fig.pdf')` |
| Word / web | SVG → PNG at 300 DPI | Use Inkscape for SVG→PNG |
| Architecture diagrams | PDF from draw.io or TikZ | Never rasterise |
| Structure renders | PNG at 600 DPI | From PyMOL/ChimeraX |

### 10.2 Naming Convention

```
fig_<chapter>_<number>_<short_description>.<ext>

Examples:
  fig_03_01_training_curves.pdf
  fig_04_02_ablation_dotplot.pdf
  fig_05_01_architecture_pipeline.pdf
  fig_05_03_attention_heatmap.png
```

### 10.3 Matplotlib Global Config

Save the following as `matplotlibrc` or load at the start of every notebook:

```python
import matplotlib as mpl

STYLE = {
    # Font
    'font.family':           'sans-serif',
    'font.sans-serif':       ['Inter', 'Helvetica Neue', 'Arial'],
    'font.size':             8,
    'axes.labelsize':        8,
    'axes.titlesize':        9,
    'xtick.labelsize':       7,
    'ytick.labelsize':       7,
    'legend.fontsize':       7.5,

    # Spines
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'axes.linewidth':        1.0,
    'axes.edgecolor':        '#37474F',

    # Grid
    'axes.grid':             True,
    'grid.color':            '#CFD8DC',
    'grid.linewidth':        0.5,
    'grid.linestyle':        '--',
    'grid.alpha':            0.7,
    'axes.axisbelow':        True,

    # Lines and markers
    'lines.linewidth':       2.0,
    'lines.markersize':      6,
    'errorbar.capsize':      3,

    # Legend
    'legend.frameon':        False,
    'legend.borderpad':      0.4,
    'legend.handlelength':   1.5,

    # Colours
    'axes.prop_cycle': mpl.cycler(color=[
        '#00BCD4',  # Teal (primary)
        '#5C6BC0',  # Slate Blue
        '#EF5350',  # Coral
        '#FFA726',  # Amber
        '#43A047',  # Green
        '#AB47BC',  # Violet
        '#78909C',  # Blue-Grey
    ]),

    # Output
    'figure.dpi':            150,   # screen
    'savefig.dpi':           300,   # file
    'savefig.bbox':          'tight',
    'savefig.pad_inches':    0.05,
    'figure.facecolor':      '#FAFAFA',
    'axes.facecolor':        '#FAFAFA',
}

mpl.rcParams.update(STYLE)
```

---

## 11. Common Mistakes to Avoid

| ❌ Avoid | ✅ Instead |
|---|---|
| Rainbow / jet colourmaps | Sequential teal or diverging blue–red palettes |
| Top and right axis spines | Remove both |
| Legend with border box | `legend.frameon = False` |
| Rotated y-axis tick labels | Abbreviate labels; keep horizontal |
| Pie charts | Bar chart or dot plot |
| 3D bar / pie charts | Never |
| Gridlines on all 4 sides | Major gridlines only, behind data |
| Bold or italic axis tick labels | Regular weight only |
| "Training Loss" as y-label | "Training loss" (sentence case) |
| Unlabelled confidence intervals | State in caption: "Shaded region = 95% CI" |
| Figures cropped too tight | ≥5% padding on all sides |
| Different fonts in different figures | Apply global rcParams across all notebooks |
| Panel labels A, B, C (caps) | (a), (b), (c) — lower case, bold |

---

## 12. Quick Checklist Before Submission

Before including any figure in the dissertation, verify:

- [ ] Top and right spines removed
- [ ] Font is consistent with global style (Inter / Helvetica Neue / Arial)
- [ ] Axis labels are sentence case with units
- [ ] Error bars / confidence bands are described in the caption
- [ ] Colour palette follows this guide
- [ ] Legend has no border; labels are unambiguous
- [ ] Panel labels are **(a)**, **(b)** style, bold
- [ ] Figure is legible at 60% zoom (print size)
- [ ] Caption follows "Fig. N | Title. (a) ... (b) ..." format
- [ ] Exported at ≥300 DPI (or as vector PDF)
- [ ] File named according to naming convention
- [ ] Human / oracle reference lines are labelled directly
- [ ] No chartjunk (shadows, gradients, unnecessary borders)

---

*Guide version 1.0 — adapt section 6 as new figure types arise.*
