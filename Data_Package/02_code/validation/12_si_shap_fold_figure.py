# -*- coding: utf-8 -*-
"""
12_si_shap_fold_figure.py
Regenerates the Supplementary Figure 'SI_Fig_SHAP_fold_stability' from the
packaged per-fold SHAP table (04_outputs/validation/shap_abs_contrib.csv),
displaying the ten dynamic drivers under their substantive labels (identical
to the manuscript text and to Figures 2/5/6).

Reads:  04_outputs/validation/shap_abs_contrib.csv
Writes: PNG (300 dpi) + PDF of the fold-stability bar chart.

Run from the Data_Package root:
   python 02_code/validation/12_si_shap_fold_figure.py [output_dir]
Default output dir: <Data_Package>/06_figures (SI copies are mirrored to
R3_Supplementary/SI_Figures by the submission workflow).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, '06_figures')
if len(sys.argv) > 1:
    OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)

LABELS = {
    'PerCapitaGDP': 'Economic Development Level',
    'SecondaryIndustryShare': 'Industrial Structure',
    'TertiaryIndustryShare': 'Economic Modernization',
    'FiscalCapacityPerCapita': 'Public Financial Resources',
    'ConstructionIntensity': 'Infrastructure Expansion',
    'UrbanizationRate': 'Urbanization Level',
    'PatentsPerCapita': 'Technological Innovation',
    'EnergyIntensity': 'Energy Efficiency',
    'PolicyCount': 'Government Attitude',
    'PublicAwarenessIndex': 'Public Environmental Awareness',
}
TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
})

df = pd.read_csv(os.path.join(ROOT, '04_outputs', 'validation', 'shap_abs_contrib.csv'))
df = df.set_index('feature')
df.index = [LABELS.get(f, f) for f in df.index]
# order: highest mean |SHAP| at the top (Energy Efficiency first, as in Fig. 5B/S7)
order = df.mean(axis=1).sort_values(ascending=False).index
df = df.loc[order]

fig, ax = plt.subplots(figsize=(12.6, 10))
n = len(df)
n_f = len(TEST_YEARS)
width = 0.8 / n_f
for j, (year, color) in enumerate(zip(TEST_YEARS, COLORS)):
    vals = df['fold%d' % (j + 1)].to_numpy()
    ax.barh(np.arange(n) + (j - (n_f - 1) / 2) * width, vals, height=width,
            color=color, edgecolor='white', linewidth=0.4, label=str(year))
ax.set_yticks(np.arange(n))
ax.set_yticklabels(list(df.index))
ax.set_xlabel('Mean |SHAP Value| (log-scale target)')
ax.set_title('SHAP Rank Stability of the Ten Dynamic Drivers Across Evaluation Windows',
             fontweight='bold', pad=14)
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--', alpha=0.35, which='major')
ax.set_axisbelow(True)
ax.legend(title='Test Year (Fold)', loc='lower right', frameon=True,
          edgecolor='#888888', framealpha=0.95)

fig.tight_layout()
name = 'SI_Fig_SHAP_fold_stability'
fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
fig.savefig(os.path.join(OUT, name + '.pdf'))
plt.close(fig)
print('written:', os.path.join(OUT, name + '.png'), '|', os.path.join(OUT, name + '.pdf'))
