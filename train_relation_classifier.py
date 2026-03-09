"""
train_relation_classifier.py

Trains a lightweight ML classifier to predict the relation type between
a 2024 and 2013 law article pair, replacing the LLM-based approach in
03_extract_comparisons.py.

Uses existing LLM-labeled data in comparisons_json/ as training data.
Encodes article pairs using the Vietnamese bi-encoder already in the project.
Feature vector: [emb_2024 ; emb_2013 ; |emb_2024 - emb_2013| ; emb_2024 * emb_2013]

Output: relation_classifier/classifier.pkl  (MLP)
        relation_classifier/label_encoder.pkl
        relation_classifier/config.json
"""

import os
import re
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

ENCODER_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'
COMPARISONS_DIR = 'comparisons_json'
CHUNKS_2024_DIR = 'chunks_2024'
CHUNKS_2013_DIR = 'chunks_2013'
MODEL_DIR = 'relation_classifier'
MAX_TEXT_CHARS = 1024

# Normalize all label variants to lowercase_no_accent form
LABEL_MAP = {
    'sua_doi_bo_sung':   'sua_doi_bo_sung',
    'SỬA_ĐỔI_BỔ_SUNG':  'sua_doi_bo_sung',
    'thay_the_hoan_toan':'thay_the_hoan_toan',
    'THAY_THẾ_HOÀN_TOÀN':'thay_the_hoan_toan',
    'giu_nguyen':        'giu_nguyen',
    'GIỮ_NGUYÊN':        'giu_nguyen',
    'dieu_luat_moi':     'dieu_luat_moi',
    'ĐIỀU_LUẬT_MỚI':     'dieu_luat_moi',
}

NO_MATCH_PLACEHOLDER = "không có điều luật tương ứng trong luật cũ"


def read_text(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()[:MAX_TEXT_CHARS]


def article_num(article_id):
    m = re.search(r'dieu_(\d+)_', str(article_id))
    return m.group(1) if m else None


def load_samples():
    samples = []
    for fname in os.listdir(COMPARISONS_DIR):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(COMPARISONS_DIR, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_label = data.get('change_type') or data.get('type', '')
        label = LABEL_MAP.get(raw_label)
        if not label:
            continue

        source_id = data.get('source_id_2024', '')
        target_id = data.get('target_id_2013')

        num24 = article_num(source_id)
        if not num24:
            continue
        text_2024 = read_text(os.path.join(CHUNKS_2024_DIR, f'dieu_{num24}_2024.txt'))
        if not text_2024:
            continue

        if target_id and str(target_id).lower() != 'null':
            num13 = article_num(target_id)
            text_2013 = read_text(os.path.join(CHUNKS_2013_DIR, f'dieu_{num13}_2013.txt')) if num13 else ""
        else:
            text_2013 = ""

        samples.append({
            'text_2024': text_2024,
            'text_2013': text_2013 if text_2013 else NO_MATCH_PLACEHOLDER,
            'label': label,
        })

    return samples


def build_features(samples, encoder):
    texts_24 = [s['text_2024'] for s in samples]
    texts_13 = [s['text_2013'] for s in samples]

    print("Encoding 2024 articles...")
    emb_24 = encoder.encode(texts_24, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    print("Encoding 2013 articles...")
    emb_13 = encoder.encode(texts_13, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    diff    = np.abs(emb_24 - emb_13)
    product = emb_24 * emb_13
    return np.concatenate([emb_24, emb_13, diff, product], axis=1)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading labeled samples from comparisons_json/...")
    samples = load_samples()
    print(f"Loaded {len(samples)} samples.")

    dist = Counter(s['label'] for s in samples)
    print(f"Label distribution: {dict(dist)}")

    if len(samples) < 10:
        print("ERROR: Not enough samples to train. Run 03_extract_comparisons.py first.")
        return

    print(f"\nLoading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    print("\nBuilding feature vectors...")
    X = build_features(samples, encoder)
    labels = [s['label'] for s in samples]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Classes: {list(le.classes_)}")

    # Stratified split; fall back to random if any class has < 2 samples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42)

    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    print("\nTraining MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, os.path.join(MODEL_DIR, 'classifier.pkl'))
    joblib.dump(le,  os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    with open(os.path.join(MODEL_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'encoder_model': ENCODER_MODEL, 'classes': list(le.classes_)}, f, indent=2)

    print(f"\nModel saved to '{MODEL_DIR}/'")
    print("Next step: run 03_extract_comparisons_ml.py to generate comparisons without LLM.")


if __name__ == '__main__':
    main()
