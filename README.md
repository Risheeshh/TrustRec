# TrustRec

Hybrid movie recommender + live shilling-attack simulator and detector (CSE4143 project).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
MovieLens 100K downloads automatically on first run into `data/ml-100k/`.
If you're offline, download https://files.grouplens.org/datasets/movielens/ml-100k.zip,
unzip into `data/`, so that `data/ml-100k/u.data` exists. (Without it, synthetic demo data is used.)

Optional: set `GEMINI_API_KEY` to enable the **CineCompass AI** movie-overview feature.
`GEMINI_MODEL` can be set to override the default Gemini model. The AI explanations
run independently and do not affect recommendation scores.

## Files
- `core.py` - recommender, attacks, detector, metrics (no UI code)
- `app.py`  - Streamlit interface

## Report notes
- Experiments to report: 3 attack types x attack sizes (1,3,5,10%) x hybrid weight alpha. Record
  precision@10, target hit ratio@10, target average rank, detector precision/recall.
- Known limitations (state them in the report): the detector is trained on simulated attacks over the
  same real users it is tested on; attacks are synthetic, not real-world fraud; only 3 attack models.
- Possible extensions: Isolation Forest (unsupervised) as a second detector; time-aware features.
