import os
import pandas as pd
import numpy as np
import yaml
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from config import Config, DATA_DIR
from utility import Dataset, get_logger
from llm_logging import set_llm_log_path, write_llm_summary
from saged_meta import train_base_classifiers, generate_meta_features_with_models, get_similarity
from zeroed_features import create_criteria_features, get_top_k_related_attributes, create_zeroed_features
from zeroed_llm import LLMAnalyzer

def load_run_config(config_path="ares-main/run_config.yaml"):
    if not os.path.exists(config_path):
        # Fallback if running from inside ares-main
        config_path = "run_config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def save_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(content))

def main():
    config = Config()

    # Setup Results Directory
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
    run_config = load_run_config()
    dataset_name = run_config.get("target_dataset", "hospital")

    # ZeroED feature settings (can be overridden in run_config.yaml)
    fasttext_model_path = run_config.get("fasttext_model_path", config.fasttext_model_path)
    config.fasttext_model_path = fasttext_model_path
    config.fasttext_dim = int(run_config.get("fasttext_dim", config.fasttext_dim))
    config.top_k_related = int(run_config.get("top_k_related", config.top_k_related))
    # ZeroED clustering hyperparameter
    config.n_clusters = int(run_config.get("n_clusters", 5))

    # ── Ablation toggles ──────────────────────────────────────────────────────
    use_history = bool(run_config.get("use_history", True))
    use_confidence = bool(run_config.get("use_confidence", True))
    ablation_variant = (
        "full_ares"    if (use_history and use_confidence) else
        "w_history"    if (use_history and not use_confidence) else
        "w_confidence" if (not use_history and use_confidence) else
        "baseline"
    )
    
    results_dir = os.path.join("ares-main", "results", f"{timestamp}_{dataset_name}_{ablation_variant}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Enable per-run LLM logging (prompt/response + approximate token usage)
    set_llm_log_path(os.path.join(results_dir, "llm_logging.txt"))

    # Subdirectories
    dirs = {
        "distri_analys": os.path.join(results_dir, "distri_analys"),
        "guide": os.path.join(results_dir, "guide"),
        "funcs": os.path.join(results_dir, "funcs"),
        "llm_label": os.path.join(results_dir, "llm_label"),
        "err_gen": os.path.join(results_dir, "err_gen"), # Placeholder if needed
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    logger = get_logger("ARES", os.path.join(results_dir, "run.log"))
    logger.info(f"Ablation variant: {ablation_variant} (use_history={use_history}, use_confidence={use_confidence})")
    
    # 1. Load Datasets
    try:
        dirty_dataset = Dataset(dataset_name)
        logger.info(f"Loaded dirty dataset: {dirty_dataset}")
    except FileNotFoundError:
        logger.error(f"Dataset {dataset_name} not found. Please ensure data is in {DATA_DIR}")
        return

    # Load historical datasets
    historical_datasets = []
    
    # Use configured historical datasets if available, else load all
    configured_historical = run_config.get("historical_datasets", [])
    
    if configured_historical:
        dataset_names = set(configured_historical)
    else:
        # Find all unique dataset names in data dir
        all_files = os.listdir(DATA_DIR)
        dataset_names = set()
        for f in all_files:
            if f.endswith("_clean.csv"):
                name = f.replace("_clean.csv", "")
                if name != dataset_name:
                    dataset_names.add(name)
    
    for name in dataset_names:
        try:
            ds = Dataset(name)
            historical_datasets.append(ds)
            logger.info(f"Loaded historical dataset: {ds}")
        except Exception as e:
            logger.warning(f"Could not load historical dataset {name}: {e}")
    
    # 2. Compute ZeroED features for the dirty dataset (value / vicinity / pattern / FastText)
    # This is done once here and reused across the SAGED meta-learning phase and the
    # final error-detection classifier, avoiding double computation.
    logger.info("Computing ZeroED features for dirty dataset (vicinity, pattern, FastText)...")
    top_k_related = get_top_k_related_attributes(dirty_dataset.dirty_df, k=config.top_k_related)
    zeroed_feats = create_zeroed_features(
        dirty_dataset.dirty_df, top_k_related,
        fasttext_model_path=config.fasttext_model_path,
        fasttext_dim=config.fasttext_dim,
        k=config.top_k_related
    )

    # 3. SAGED Phase — train base classifiers on historical datasets with ZeroED features
    # and generate meta-features (base-classifier probability predictions) for dirty data.

    logger.info("Starting SAGED Phase...")
    if not use_history:
        logger.info("[Ablation] use_history=False — skipping historical meta-features (SAGED phase).")
        meta_features = {col: pd.DataFrame() for col in dirty_dataset.dirty_df.columns}
    elif historical_datasets:
        base_models = train_base_classifiers(
            historical_datasets,
            fasttext_model_path=config.fasttext_model_path,
            fasttext_dim=config.fasttext_dim,
            k=config.top_k_related
        )
        similarity = get_similarity(dirty_dataset, historical_datasets, config)
        # Pass pre-computed dirty zeroed_feats to avoid re-computation
        meta_features = generate_meta_features_with_models(
            dirty_dataset, base_models, similarity, config,
            dirty_feats=zeroed_feats
        )
    else:
        logger.warning("No historical datasets found. Skipping meta-features.")
        meta_features = {col: pd.DataFrame() for col in dirty_dataset.dirty_df.columns}

    # 4. ZeroED Phase — LLM-driven distribution analysis, guidelines, and criteria functions.
    logger.info("Starting ZeroED Phase...")
    llm_analyzer = LLMAnalyzer(dirty_dataset.dirty_df)
    
    criteria_features_dict = {}
    verified_labels = {} # {col: Series of labels (0, 1, NaN)}

    # Use a fixed list of columns to avoid issues if LLM code modifies the dataframe
    columns_to_process = list(dirty_dataset.dirty_df.columns)
    
    for col in columns_to_process:
        logger.info(f"Processing column: {col}")
        
        # LLM Analysis
        dist_analysis, da_prompt, da_resp = llm_analyzer.generate_distribution_analysis(col)
        save_text(os.path.join(dirs["distri_analys"], f"distri_analys_{col}.txt"), dist_analysis)
        save_text(os.path.join(dirs["distri_analys"], f"prompt_distri_analys_{col}.txt"), da_prompt)
        
        # Guidelines
        guidelines, g_prompt = llm_analyzer.generate_guidelines(col, dist_analysis)
        save_text(os.path.join(dirs["guide"], f"guide_{col}.txt"), guidelines)
        save_text(os.path.join(dirs["guide"], f"prompt_{col}.txt"), g_prompt)
        
        # Error Generation (for analysis/augmentation)
        err_gen_resp, eg_prompt = llm_analyzer.generate_errors(col)
        save_text(os.path.join(dirs["err_gen"], f"err_gen_{col}.txt"), err_gen_resp)
        save_text(os.path.join(dirs["err_gen"], f"prompt_err_gen_{col}.txt"), eg_prompt)

        # Criteria Functions
        funcs_code, f_prompt, f_resp = llm_analyzer.generate_criteria_functions(col, guidelines)
        save_text(os.path.join(dirs["funcs"], f"funcs_{col}.txt"), "\n\n".join(funcs_code))
        save_text(os.path.join(dirs["funcs"], f"prompt_funcs_{col}.txt"), f_prompt)
        
        # Labeling (Clustering + LLM)
        # Cluster the column values using ZeroED features
        col_feats = zeroed_feats.get(col, pd.DataFrame(index=dirty_dataset.dirty_df.index))
        col_feat_arr = col_feats.fillna(0).values
        n_clusters = min(config.n_clusters, len(col_feat_arr))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(col_feat_arr)

        # Select representatives: the actual data point closest to each cluster centroid
        # (mirrors ZeroED's pairwise_distances_argmin_min strategy — deterministic, no randomness)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, col_feat_arr)
        samples = []
        for idx in closest:
            row_dict = dirty_dataset.dirty_df.iloc[idx].to_dict()
            samples.append((idx, row_dict))
                
        # LLM Labeling
        llm_labels, l_prompt, l_resp = llm_analyzer.label_samples(samples, col, guidelines, use_confidence=use_confidence)
        save_text(os.path.join(dirs["llm_label"], f"llm_label_{col}.txt"), str(llm_labels))
        save_text(os.path.join(dirs["llm_label"], f"prompt_label_{col}.txt"), l_prompt)
        
        # Label Propagation
        # Assign cluster label based on representative
        propagated_labels = pd.Series(np.nan, index=dirty_dataset.dirty_df.index)
        for idx, label in llm_labels.items():
            c_id = clusters[idx]
            propagated_labels[clusters == c_id] = label
            
        # Mutual Verification (Filter criteria functions)
        # Check if functions agree with propagated labels
        valid_funcs = []
        for func in funcs_code:
            # Execute func on labeled data
            # If agreement is high, keep func
            valid_funcs.append(func) # Keep all for now in this skeleton
            
        # Generate Criteria Features
        c_feats = create_criteria_features(dirty_dataset.dirty_df[[col]], {col: valid_funcs})
        criteria_features_dict[col] = c_feats.get(col, pd.DataFrame())
        
        verified_labels[col] = propagated_labels



    # 5. Combine Features & Train
    logger.info("Training Final Models...")
    
    final_predictions = pd.DataFrame(index=dirty_dataset.dirty_df.index, columns=columns_to_process)
    
    for col in columns_to_process:
        # ZeroED base features: value / vicinity / pattern / FastText
        if col not in zeroed_feats:
            logger.warning(f"ZeroED features missing for column {col}; using empty placeholder")
        f_zeroed = zeroed_feats.get(col, pd.DataFrame(index=dirty_dataset.dirty_df.index))

        # SAGED meta-features: base-classifier probability predictions
        if col not in meta_features:
            logger.debug(f"Meta features missing for column {col}; using empty placeholder")
        f_meta = meta_features.get(col, pd.DataFrame(index=dirty_dataset.dirty_df.index))

        # LLM criteria features
        if col not in criteria_features_dict:
            logger.debug(f"Criteria features missing for column {col}; using empty placeholder")
        f_criteria = criteria_features_dict.get(col, pd.DataFrame(index=dirty_dataset.dirty_df.index))

        # Combine: ZeroED base + SAGED meta + LLM criteria
        X = pd.concat([f_zeroed, f_meta, f_criteria], axis=1).fillna(0)
        
        # Labels
        y = verified_labels[col]
        
        # Train on labeled data
        mask = y.notna()
        if mask.sum() > 0:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
            clf.fit(X[mask], y[mask])
            if hasattr(clf, "classes_") and len(clf.classes_) > 1:
                probs = clf.predict_proba(X)[:, 1]
                preds = (probs > 0.6).astype(int)
            else:
                preds = clf.predict(X)
            final_predictions[col] = preds
        else:
            # ── Fallback cascade: generate pseudo-labels, then train the same RF ──
            # Using pseudo-labels + classifier is far better than plain thresholding
            # because the learned boundary exploits the full ZeroED feature space.
            from sklearn.ensemble import RandomForestClassifier, IsolationForest

            pseudo_labels = None
            pseudo_source = None

            # A: meta-features from historical classifiers (most reliable signal)
            if not f_meta.empty and 'meta_mean' in f_meta.columns:
                meta_mean = f_meta['meta_mean'].fillna(0)
                # Only use high-confidence pseudo-labels to avoid noise
                high_conf_mask = (meta_mean > 0.65) | (meta_mean < 0.20)
                if high_conf_mask.sum() >= 10:
                    pseudo_labels = (meta_mean > 0.65).astype(int)
                    pseudo_labels = pseudo_labels[high_conf_mask]
                    pseudo_source = "meta-feature pseudo-labels"

            # B: criteria functions — use them as noisy labels but only where they agree
            if pseudo_labels is None:
                f_criteria_col = criteria_features_dict.get(col, pd.DataFrame())
                if not f_criteria_col.empty:
                    # Treat -1 (exec error) as abstain, 0/1 as votes
                    valid = f_criteria_col.replace(-1, np.nan)
                    vote_mean = valid.mean(axis=1)   # NaN where all functions failed
                    # High consensus threshold: all available functions agree
                    high_agree = vote_mean.notna() & ((vote_mean >= 0.8) | (vote_mean <= 0.2))
                    if high_agree.sum() >= 10:
                        pseudo_labels = (vote_mean[high_agree] > 0.5).astype(int)
                        pseudo_source = "criteria-function consensus pseudo-labels"

            # Train RF on pseudo-labels if we have enough and both classes are present
            if pseudo_labels is not None and pseudo_labels.nunique() > 1:
                logger.warning(f"No labels for '{col}': training on {pseudo_source} ({mask.sum()} → {len(pseudo_labels)} pseudo-labeled rows).")
                X_pseudo = X.loc[pseudo_labels.index].fillna(0)
                clf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
                clf.fit(X_pseudo, pseudo_labels)
                if len(clf.classes_) > 1:
                    probs = clf.predict_proba(X.fillna(0))[:, 1]
                    preds = (probs > 0.55).astype(int)  # Slightly lower threshold for pseudo-trained model
                else:
                    preds = clf.predict(X.fillna(0))
                final_predictions[col] = preds

            # C: unsupervised IsolationForest on ZeroED features (last resort)
            elif not f_zeroed.empty:
                logger.warning(f"No labels for '{col}': using IsolationForest (unsupervised).")
                iso = IsolationForest(contamination=0.1, random_state=42)
                raw = iso.fit_predict(f_zeroed.fillna(0))  # -1 = anomaly, 1 = normal
                final_predictions[col] = (raw == -1).astype(int)

            else:
                logger.warning(f"No labels and no fallback signal for '{col}': predicting all clean.")
                final_predictions[col] = 0

    # 6. Evaluate
    actual_errors = dirty_dataset.get_actual_errors()
    
    # Align columns between predictions and actual errors
    common_cols = final_predictions.columns.intersection(actual_errors.columns)
    y_true = actual_errors[common_cols].values.flatten()
    y_pred = final_predictions[common_cols].values.flatten().astype(int)
    
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    logger.info(f"Final Results - F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Persist metrics as a small CSV so the ablation runner can collect them
    metrics_path = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame([{
        "dataset": dataset_name,
        "variant": ablation_variant,
        "use_history": use_history,
        "use_confidence": use_confidence,
        "n_clusters": config.n_clusters,
        "top_k_related": config.top_k_related,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }]).to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    # Write ZeroED-compatible LLM token summary for run-to-run comparison
    write_llm_summary()
    logger.info(f"LLM token summary saved to {os.path.join(results_dir, 'llm_logging.txt')}")

if __name__ == "__main__":
    main()
