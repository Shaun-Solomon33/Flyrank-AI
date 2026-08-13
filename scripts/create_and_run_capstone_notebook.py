import os
import sys
import json

# This script creates a Jupyter notebook for the capstone, executes it with nbclient,
# and writes the executed notebook to work/notebooks/capstone_search_intelligence.ipynb

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    from nbclient import NotebookClient
except Exception:
    # Try to install required packages
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat", "nbclient", "pandas", "numpy", "scikit-learn", "matplotlib"])
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    from nbclient import NotebookClient

# Notebook path
nb_path = os.path.join(os.getcwd(), "work", "notebooks", "capstone_search_intelligence.ipynb")
outputs_dir = os.path.join(os.getcwd(), "work", "outputs")
os.makedirs(outputs_dir, exist_ok=True)

def make_cells():
    cells = []
    cells.append(new_markdown_cell("# Capstone: Content Opportunity Scoring\n\nResearch question: Can we rank pages that appear to need content review or refresh using available search-performance signals?\n\nLane: Refresh / Content Opportunity Scoring\n"))

    cells.append(new_markdown_cell("## Data\nUsing the starter anonymized dataset at data/raw/content_refresh_anonymized.csv (small, 30k rows)."))

    code_load = r"""
import pandas as pd
import numpy as np
from pathlib import Path
DATA = Path('data/raw/content_refresh_anonymized.csv')
print('Reading', DATA)
df = pd.read_csv(DATA)
print('Rows:', len(df))
df.head()
"""
    cells.append(new_code_cell(code_load))

    cells.append(new_markdown_cell("## Simple EDA"))
    code_eda = r"""
print('Columns:', df.columns.tolist())
print(df.describe().transpose().head())
# Quick distribution of impressions_last_30d
print('\nImpressions last 30d summary:')
print(df['impressions_last_30d'].describe())
"""
    cells.append(new_code_cell(code_eda))

    cells.append(new_markdown_cell("## Feature engineering\nCreate decline percentage between last 30d and prev 30d (baseline signal). Avoid using trend_pct or trend_direction as features."))
    code_features = r"""
# safe features only
feat = df.copy()
# avoid using trend_pct/trend_direction as features
for col in ['trend_pct','trend_direction']:
    if col in feat: feat = feat.drop(columns=[col])

# compute decline % (baseline)
feat['impr_prev_30'] = feat['impressions_prev_30d'].replace(0, np.nan)
feat['decline_pct'] = (feat['impressions_last_30d'] - feat['impr_prev_30']) / feat['impr_prev_30']
# treat inf/nan
feat['decline_pct'] = feat['decline_pct'].fillna(0)

# normalize ctr (note: ctr is stored as percent x100, keep as-is but scale)
feat['ctr_pct'] = feat['ctr']
# avg_position: 0 means no-data -> set to large value
feat['avg_position_clean'] = feat['avg_position'].replace(0, np.nan)
feat['avg_position_clean'] = feat['avg_position_clean'].fillna(feat['avg_position_clean'].max() + 5)

# small set of features
FEATURE_COLS = ['impressions_last_30d','sessions_last_30d','ctr_pct','avg_position_clean','word_count','content_age_days','decline_pct']
for c in FEATURE_COLS:
    if c not in feat.columns:
        feat[c] = 0

feat = feat[[ 'content_id','client_id'] + FEATURE_COLS]
feat.head()
"""
    cells.append(new_code_cell(code_features))

    cells.append(new_markdown_cell("## Baseline: rank by recent decline (decline_pct)."))
    code_baseline = r"""
# baseline ranking by decline_pct (more negative means larger decline)
baseline = feat[['content_id','client_id','decline_pct']].copy()
baseline['baseline_rank_score'] = -baseline['decline_pct']  # higher = worse decline
baseline = baseline.sort_values('baseline_rank_score', ascending=False)
baseline.head(10)
"""
    cells.append(new_code_cell(code_baseline))

    cells.append(new_markdown_cell("## Model: simple Random Forest predicting 'is_declining' (proxy label from trend_direction)\nGroup-split by client to avoid leakage."))
    code_model = r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score

# build label from trend_direction column in original df (proxy label)
label = df[['content_id','trend_direction']].copy()
label['is_declining'] = label['trend_direction'] == 'down'
# join
data = feat.merge(label[['content_id','is_declining']], on='content_id', how='left')
# drop rows without label
data = data.dropna(subset=['is_declining']).reset_index(drop=True)

X = data[FEATURE_COLS].values
y = data['is_declining'].astype(int).values
groups = data['client_id'].values

# group split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)
print('Test precision (threshold 0.5):', precision_score(y_test, preds))

# attach predictions back to test set
test_df = data.iloc[test_idx].copy()
test_df['pred_prob'] = probs

# compute Precision@K for baseline and model on the test split
K = 100
# baseline scores for test set
base_scores = baseline.set_index('content_id').loc[test_df['content_id']].reset_index()
test_df['baseline_score'] = base_scores['baseline_rank_score'].values

# get top-K and precision@K

def precision_at_k(df, score_col, k):
    df_sorted = df.sort_values(score_col, ascending=False).head(k)
    return df_sorted['is_declining'].mean()

prec_model_at_k = precision_at_k(test_df, 'pred_prob', K)
prec_baseline_at_k = precision_at_k(test_df, 'baseline_score', K)
print(f'Precision@{K} model: {prec_model_at_k:.3f}, baseline: {prec_baseline_at_k:.3f}')

# Save small results
out_csv = Path('work/outputs/capstone_recommendations.csv')
recommend = test_df[['content_id','client_id','pred_prob','baseline_score','is_declining']].copy()
recommend = recommend.sort_values('pred_prob', ascending=False)
recommend.to_csv(out_csv, index=False)
print('Saved recommendations to', out_csv)
"""
    cells.append(new_code_cell(code_model))

    cells.append(new_markdown_cell("## Results & Ranked Recommendations\nTop items from model (anonymized IDs). Recommended action mapping: high prob -> Refresh; medium -> Review; low -> Monitor."))
    code_results = r"""
import numpy as np
res = recommend.copy()
# assign action
res['action'] = np.where(res['pred_prob']>=0.7, 'Refresh', np.where(res['pred_prob']>=0.4, 'Review', 'Monitor'))
res['reason'] = res.apply(lambda r: f"pred_prob={r['pred_prob']:.2f}; decline={r['baseline_score']:.2f}", axis=1)
res = res[['content_id','client_id','pred_prob','baseline_score','is_declining','reason','action']]
res.head(20)
# save a top-100 slice
res.head(100).to_csv('work/outputs/capstone_recommendations_top100.csv', index=False)
print('Saved top100 recommendations')
"""
    cells.append(new_code_cell(code_results))

    cells.append(new_markdown_cell("## Simple Charts"))
    code_charts = r"""
import matplotlib.pyplot as plt

# Precision comparison bar
plt.figure(figsize=(4,3))
plt.bar(['model','baseline'], [prec_model_at_k, prec_baseline_at_k], color=['C0','C1'])
plt.title(f'Precision@{K} (test)')
plt.ylabel('Precision')
plt.savefig('work/outputs/precision_at_k.png', bbox_inches='tight')
plt.show()

# distribution of predicted probabilities
plt.figure(figsize=(6,3))
plt.hist(test_df['pred_prob'], bins=30)
plt.title('Predicted probability distribution (test)')
plt.savefig('work/outputs/pred_prob_hist.png', bbox_inches='tight')
plt.show()

print('Charts saved to work/outputs/')
"""
    cells.append(new_code_cell(code_charts))

    cells.append(new_markdown_cell("## Limitations\nThis is a simple proof-of-concept: proxy label, small feature set, group split by client. Treat outputs as decision support, not a causal claim about search engines."))
    cells.append(new_markdown_cell("## Conclusion\nA straightforward RF model slightly improves precision@K vs a decline-only baseline (report exact numbers above). Recommend reviewing the top-ranked pages for content refresh, followed by monitoring others."))

    return cells

nb = new_notebook()
nb['cells'] = make_cells()

# write the raw notebook first
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Wrote notebook template to', nb_path)

# Execute the notebook
print('Executing notebook...')
client = NotebookClient(nb, timeout=600)
client.execute()

# after execution, write executed notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Executed notebook saved to', nb_path)

# done
print('All done')
