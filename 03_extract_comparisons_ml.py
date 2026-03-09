"""
03_extract_comparisons_ml.py

ML-based replacement for 03_extract_comparisons.py.

For each 2024 article:
  1. Finds top-K candidate 2013 articles by cosine similarity (bi-encoder).
  2. Runs the trained MLP classifier on each (2024, 2013) pair to predict
     the relation type and confidence.
  3. Picks the candidate + relation with the highest classifier confidence.
  4. Writes the result to comparisons_json/ in the same format as before.

Requires: relation_classifier/ directory produced by train_relation_classifier.py
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

CHUNKS_2024_DIR  = 'chunks_2024'
CHUNKS_2013_DIR  = 'chunks_2013'
OUTPUT_DIR       = 'comparisons_json'
CLASSIFIER_DIR   = 'relation_classifier'
TOP_K_CANDIDATES = 10
MAX_TEXT_CHARS   = 1024
NO_MATCH_PLACEHOLDER = "không có điều luật tương ứng trong luật cũ"


def read_text(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()[:MAX_TEXT_CHARS]


def article_num(fname_or_id):
    m = re.search(r'dieu_(\d+)_', str(fname_or_id))
    return m.group(1) if m else None


def pair_features(emb_24, emb_13):
    diff    = np.abs(emb_24 - emb_13)
    product = emb_24 * emb_13
    return np.concatenate([emb_24, emb_13, diff, product]).reshape(1, -1)


def load_classifier():
    clf_path = os.path.join(CLASSIFIER_DIR, 'classifier.pkl')
    le_path  = os.path.join(CLASSIFIER_DIR, 'label_encoder.pkl')
    cfg_path = os.path.join(CLASSIFIER_DIR, 'config.json')

    if not all(os.path.exists(p) for p in [clf_path, le_path, cfg_path]):
        raise FileNotFoundError(
            f"Model files not found in '{CLASSIFIER_DIR}/'. "
            "Please run train_relation_classifier.py first."
        )

    clf = joblib.load(clf_path)
    le  = joblib.load(le_path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    return clf, le, cfg


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load trained model
    print("Loading trained classifier...")
    clf, le, cfg = load_classifier()
    encoder_model = cfg['encoder_model']
    classes = list(le.classes_)
    new_article_class = 'dieu_luat_moi'

    print(f"Loading encoder: {encoder_model}")
    encoder = SentenceTransformer(encoder_model)

    # Load and encode all 2013 articles once
    print("\nLoading and encoding all 2013 articles...")
    files_2013 = sorted(f for f in os.listdir(CHUNKS_2013_DIR) if f.endswith('.txt'))
    ids_2013, texts_2013 = [], []
    for fname in files_2013:
        num = article_num(fname)
        if not num:
            continue
        text = read_text(os.path.join(CHUNKS_2013_DIR, fname))
        if text:
            ids_2013.append(f'dieu_{num}_2013')
            texts_2013.append(text)

    embs_2013 = encoder.encode(texts_2013, normalize_embeddings=True,
                                show_progress_bar=True, batch_size=32)

    # Encode the "no match" placeholder for new-article detection
    no_match_emb = encoder.encode([NO_MATCH_PLACEHOLDER],
                                   normalize_embeddings=True)[0]

    # Process each 2024 article
    files_2024 = sorted(f for f in os.listdir(CHUNKS_2024_DIR) if f.endswith('.txt'))
    skipped, written = 0, 0

    print(f"\nClassifying {len(files_2024)} articles from 2024 law...")
    for fname in tqdm(files_2024):
        num24 = article_num(fname)
        if not num24:
            continue

        article_id  = f'dieu_{num24}_2024'
        output_path = os.path.join(OUTPUT_DIR, fname.replace('.txt', '.json'))

        if os.path.exists(output_path):
            skipped += 1
            continue

        text_2024 = read_text(os.path.join(CHUNKS_2024_DIR, fname))
        if not text_2024:
            continue

        emb_24 = encoder.encode([text_2024], normalize_embeddings=True)[0]

        # --- Find top-K candidates from 2013 ---
        sims      = cosine_similarity([emb_24], embs_2013)[0]
        top_k_idx = np.argsort(sims)[::-1][:TOP_K_CANDIDATES]

        best_result     = None
        best_confidence = -1.0

        for idx in top_k_idx:
            features = pair_features(emb_24, embs_2013[idx])
            proba    = clf.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba))
            pred_label = classes[pred_idx]
            confidence = float(proba[pred_idx])

            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    'source_id_2024': article_id,
                    'target_id_2013': ids_2013[idx],
                    'change_type':    pred_label,
                    'confidence':     round(confidence, 4),
                }

        # --- Also evaluate "new article" scenario ---
        features_new  = pair_features(emb_24, no_match_emb)
        proba_new     = clf.predict_proba(features_new)[0]
        if new_article_class in classes:
            new_conf = float(proba_new[classes.index(new_article_class)])
            if new_conf > best_confidence:
                best_result = {
                    'source_id_2024': article_id,
                    'target_id_2013': None,
                    'change_type':    new_article_class,
                    'confidence':     round(new_conf, 4),
                }

        if best_result:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(best_result, f, ensure_ascii=False, indent=2)
            written += 1

    print(f"\nDone. Written: {written}  Skipped (already existed): {skipped}")
    print("Next step: run 04_0_merge_json.py to continue the pipeline.")


if __name__ == '__main__':
    main()
