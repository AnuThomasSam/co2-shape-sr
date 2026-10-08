# %%
import time 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import ( roc_auc_score, classification_report, RocCurveDisplay ) 
from sklearn.model_selection import ( cross_val_score, StratifiedKFold, cross_val_predict )
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# %%
# Load the clustered dataset

data_clustered = pd.read_csv(
    "../data/processed/vehicle_co2_technology_clusters.csv"
)

print("Dataset shape:", data_clustered.shape)
print("\nColumns:") 
print(data_clustered.columns.tolist())

# %%
# BASIC DATA CHECK

print("Columns:")
print(data_clustered.columns.tolist())

print("\nDataset shape:")
print(data_clustered.shape)

print("\nMissing values:")
print(data_clustered.isnull().sum())

# %%
print("\nNumber of technology clusters:") 
print(data_clustered["technology_cluster"].nunique()) 
print("\nCluster counts:") 
print( data_clustered["technology_cluster"] .value_counts() .sort_index() )

# %%
# PROFILE ALL TECHNOLOGY CLUSTERS
cluster_profile = (
    data_clustered .groupby("technology_cluster") 
    .agg( 
        count=("technology_cluster", "size"), 
        mean_engine_size=("engine_size", "mean"), 
        mean_cylinders=("cylinders", "mean"), 
        mean_transmission_gears=("transmission_gears", "mean") 
        ) .round(3) 
    )
print("ALL TECHNOLOGY CLUSTER PROFILES")
print( cluster_profile.to_string() )

# %%
# CALCULATE TECHNOLOGY SCORE
# We create an engineering-based score for each cluster. 
# Direction: Engine size -> larger = older, Cylinders -> larger = older, Transmission gears -> larger = newer
# Therefore: Older technology score = + engine size + cylinders - transmission gears
# Because these variables have different scales, we standardise the cluster means before combining them.

score_features = [ 
    "mean_engine_size", 
    "mean_cylinders", 
    "mean_transmission_gears"
] 

cluster_score_data = cluster_profile[score_features].copy() 
# Standardise the three cluster-level engineering characteristics 
score_means = cluster_score_data.mean() 
score_stds = cluster_score_data.std(ddof=0) 
cluster_z = ( cluster_score_data - score_means ) / score_stds 
# Older technology: 
# larger engine -> positive 
# more cylinders -> positive 
# more gears -> negative 
cluster_profile["technology_score"] = ( cluster_z["mean_engine_size"] + cluster_z["mean_cylinders"] - cluster_z["mean_transmission_gears"] ) 
cluster_profile["technology_score"] = ( cluster_profile["technology_score"] .round(4) )

# %%
# RANK ALL CLUSTERS FROM OLDER TO NEWER
cluster_ranking = ( 
    cluster_profile .sort_values( 
        "technology_score", ascending=False 
        ) .copy() 
    ) 
cluster_ranking["technology_rank"] = ( np.arange(1, len(cluster_ranking) + 1) ) 
print("TECHNOLOGY CLUSTER RANKING")
print( 
    cluster_ranking[ 
        [ "count", "mean_engine_size", "mean_cylinders", "mean_transmission_gears", "technology_score", "technology_rank" ] 
    ].to_string() 
) 
print("\nInterpretation:") 
print("Rank 1 = oldest technology group") 
print( f"Rank {len(cluster_ranking)} = newest technology group" )

# %%
# DIVIDE ALL CLUSTERS INTO OLDER AND NEWER GROUPS
# With 8 clusters: 
# Older = first 4 clusters
# Newer = last 4 clusters
# This is an operational choice for converting the 8-cluster solution into
# the binary S_old / S_new temporal groups required for temporal evaluation.

number_of_clusters = len(cluster_ranking)
if number_of_clusters < 2:
    raise ValueError("At least two technology clusters are required.")
if number_of_clusters % 2 != 0:
    raise ValueError("The number of clusters is odd. " "Please define an explicit boundary between older and newer groups.")
number_per_group = number_of_clusters // 2
older_clusters = (cluster_ranking.head(number_per_group).index.tolist())
newer_clusters = (cluster_ranking.tail(number_per_group).index.tolist())
print("OLDER / NEWER TECHNOLOGY GROUPS")
print(f"\nOlder clusters ({len(older_clusters)}): {older_clusters}")
print(f"Newer clusters ({len(newer_clusters)}): {newer_clusters}")

# %%
# ADD TEMPORAL GROUP LABEL

data_adv = data_clustered.copy()
data_adv["technology_group"] = np.where(
    data_adv["technology_cluster"].isin(older_clusters), "S_old", "S_new"
)
print("TEMPORAL GROUP COUNTS")
print(data_adv["technology_group"].value_counts())

# %%
# CREATE S_old AND S_new
S_old = data_adv[ data_adv["technology_group"] == "S_old" ].copy() 
S_new = data_adv[ data_adv["technology_group"] == "S_new" ].copy() 
print("\nS_old observations:", len(S_old)) 
print("S_new observations:", len(S_new)) 
print("\nS_old clusters:") 
print( sorted(S_old["technology_cluster"].unique()) ) 
print("\nS_new clusters:") 
print( sorted(S_new["technology_cluster"].unique()) ) 

# %%
# CREATE ADVERSARIAL DATASET
# Binary labels: 0 = S_old, 1 = S_new
# The classifier will try to distinguish older from newer technology observations.

S_old["adv_label"] = 0
S_new["adv_label"] = 1
adv_data = pd.concat([S_old, S_new], ignore_index=True)
print("\n" + "=" * 70)
print("ADVERSARIAL DATASET")
print("=" * 70)
print(f"\nTotal observations: {len(adv_data)}")
print("\nLabel distribution:")
print(adv_data["adv_label"].value_counts().sort_index())

# %%
# LEAKAGE-FREE ADVERSARIAL FEATURS
numeric_features = [
    "engine_size",
    "cylinders",
    "engine_per_cylinder",
    "engine_x_cylinders",
    "engine_size_sq",
    "cylinders_sq"
]

categorical_features = [
    "make",
    "vehicle_class",
    "transmission_type",
    "fuel_type",
    "model"
]

required_features = numeric_features + categorical_features

missing_features = [
    feature for feature in required_features
    if feature not in adv_data.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("ADVERSARIAL FEATURES")

print("\nNumeric:")
print(numeric_features)

print("\nCategorical:")
print(categorical_features)

# %%
# PREPARE ADVERSARIAL DATA

X_adv = adv_data[required_features].copy()
y_adv = adv_data["adv_label"].copy()

print("ADVERSARIAL DATA")
print(f"\nFeature matrix shape: {X_adv.shape}")
print(f"Number of observations: {len(X_adv)}")
print(f"Number of labels: {y_adv.nunique()}")

# %%
# PREPROCESSING + LOGISTIC REGRESSION PIPELINE

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

adversarial_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
    )
])

# %%
# 5-FOLD CROSS-VALIDATED ADVERSARIAL VALIDATION
cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=42 ) 
start_time = time.time() 
cv_auc_scores = cross_val_score( adversarial_pipeline, X_adv, y_adv, cv=cv, scoring="roc_auc" ) 
training_time = time.time() - start_time 
auc_mean = cv_auc_scores.mean() 
auc_std = cv_auc_scores.std() 
print("ADVERSARIAL VALIDATION RESULTS") 
print( "\n5-fold ROC-AUC scores:" )
print( np.round(cv_auc_scores, 4) )
print( f"\nMean ROC-AUC: {auc_mean:.4f}" ) 
print( f"ROC-AUC Std: {auc_std:.4f}" ) 
print( f"Training time: {training_time:.4f} seconds" )

# %%
# DRIFT VERDICT
# Proposal threshold: 
# AUC >= 0.60 -> measurable distribution shift 
# AUC < 0.60 -> insufficient evidence of measurable shift

DRIFT_THRESHOLD = 0.60 
if auc_mean >= DRIFT_THRESHOLD: 
    verdict = ( "Measurable distribution shift detected." ) 
    proceed = True 
else: 
    verdict = ( "Insufficient evidence of measurable distribution shift." ) 
    proceed = False 
print("DRIFT VERDICT")
print( f"\nMean ROC-AUC: {auc_mean:.4f}" ) 
print( f"Threshold: {DRIFT_THRESHOLD:.2f}" ) 
print( f"Verdict: {verdict}" ) 
print( f"Proceed with temporal evaluation: {proceed}" )

# %%
# CROSS-VALIDATED PREDICTIONS
# Generate out-of-fold probabilities for the ROC curve.
y_adv_proba_cv = cross_val_predict(
    adversarial_pipeline,
    X_adv,
    y_adv,
    cv=cv,
    method="predict_proba"
)[:, 1] 
cv_auc_out_of_fold = roc_auc_score( y_adv, y_adv_proba_cv ) 
print("OUT-OF-FOLD ROC-AUC") 
print( f"\nOut-of-fold ROC-AUC: " f"{cv_auc_out_of_fold:.4f}" )

# %%
# CROSS-VALIDATED CLASSIFICATION REPORT
y_adv_pred_cv = ( y_adv_proba_cv >= 0.50 ).astype(int) 
print("CLASSIFICATION REPORT") 
print( classification_report( y_adv, y_adv_pred_cv, target_names=[ "S_old", "S_new" ], zero_division=0 ) )

# %%
# ROC CURVE
fig, ax = plt.subplots( figsize=(8, 6) ) 
RocCurveDisplay.from_predictions( y_adv, y_adv_proba_cv, name=f"Logistic Regression (AUC = {cv_auc_out_of_fold:.3f})", ax=ax ) 
ax.plot( [0, 1], [0, 1], linestyle="--", linewidth=1, label="Random (AUC = 0.500)" ) 
ax.set_xlabel( "False Positive Rate" ) 
ax.set_ylabel( "True Positive Rate" ) 
ax.set_title( "Adversarial Validation ROC Curve\n" "S_old vs S_new" ) 
ax.legend( loc="lower right" ) 
plt.tight_layout() 
plt.savefig( "../results/figures/adversarial_validation_roc_curve.pdf", format="pdf", bbox_inches="tight" ) 
plt.show() 
plt.close()

# %%
# CROSS-VALIDATION AUC PLOT
fig, ax = plt.subplots( figsize=(8, 5) ) 
fold_labels = [ f"Fold {i + 1}" for i in range(len(cv_auc_scores)) ] 
bars = ax.bar( fold_labels, cv_auc_scores ) 
ax.axhline( auc_mean, linestyle="--", linewidth=2, label=f"Mean AUC = {auc_mean:.4f}" ) 
ax.axhline( DRIFT_THRESHOLD, linestyle=":", linewidth=1.5, label="Drift threshold = 0.60" ) 
ax.axhline( 0.50, linestyle=":", linewidth=1, label="Random = 0.50" ) 
for bar, score in zip( bars, cv_auc_scores ): ax.text( bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.3f}", ha="center", va="bottom" ) 
ax.set_xlabel( "Cross-Validation Fold" ) 
ax.set_ylabel( "ROC-AUC" ) 
ax.set_title( "Adversarial Validation — 5-Fold ROC-AUC" ) 
ax.legend( loc="lower right" ) 
plt.tight_layout() 
plt.savefig( "../results/figures/adversarial_validation_cv_auc.pdf", format="pdf", bbox_inches="tight" ) 
plt.show() 
plt.close()

# %%
# TECHNOLOGY RANKING PLOT
# This shows how the eight clusters were ordered from older to newer.

plot_ranking = cluster_ranking.sort_values( "technology_rank" ) 
labels = [ f"Cluster {cluster}" for cluster in plot_ranking.index ] 
fig, ax = plt.subplots( figsize=(10, 6) ) 
bars = ax.bar( labels, plot_ranking["technology_score"] ) 
ax.axhline( 0, linestyle="--", linewidth=1 ) 
ax.set_xlabel( "Technology Cluster" ) 
ax.set_ylabel( "Technology Score" ) 
ax.set_title( "Technology Cluster Ranking: Older to Newer" ) 
plt.xticks( rotation=45 ) 
plt.tight_layout() 
plt.savefig( "../results/figures/technology_cluster_ordering.pdf", format="pdf", bbox_inches="tight" ) 
plt.show() 
plt.close()

# %%
# COMPARE S_old AND S_new
comparison = ( 
    data_adv 
    .groupby("technology_group") 
    .agg( 
        count=("technology_group", "size"), 
        mean_engine_size=("engine_size", "mean"), 
        mean_cylinders=("cylinders", "mean"), 
        mean_transmission_gears=("transmission_gears", "mean"), 
        mean_co2=("co2", "mean") ) 
        .round(3) 
    ) 
print("S_old vs S_new COMPARISON") 
print( comparison.to_string() )

# %%
# Print final decision

# SAVE TEMPORAL TRAINING AND TESTING DATA 
S_old_final = data_adv[ data_adv["technology_group"] == "S_old" ].copy() 
S_new_final = data_adv[ data_adv["technology_group"] == "S_new" ].copy() 
# Remove the temporary adversarial label before saving 
temporary_columns = [ "adv_label", "technology_group" ] 
S_old_final.drop( columns=temporary_columns, errors="ignore", inplace=True ) 
S_new_final.drop( columns=temporary_columns, errors="ignore", inplace=True )
# Save older technology group 
S_old_final.to_csv( "../data/processed/S_old_temporal_train.csv", index=False ) 
# Save newer technology group 
S_new_final.to_csv( "../data/processed/S_new_temporal_test.csv", index=False ) 
print("TEMPORAL DATASETS SAVED") 
print( f"\nS_old observations: {len(S_old_final)}" ) 
print( f"S_new observations: {len(S_new_final)}" ) 
print( "\nS_old clusters:" ) 
print( sorted( S_old_final["technology_cluster"].unique() ) ) 
print( "\nS_new clusters:" ) 
print( sorted( S_new_final["technology_cluster"].unique() ) ) 
print( "\nFiles saved successfully." )

# SAVE CLUSTER RANKING 
cluster_ranking.to_csv( "../results/tables/technology_cluster_ordering.csv" ) 
print( "\nTechnology cluster ordering saved." )

# SAVE ADVERSARIAL VALIDATION SUMMARY 
summary = pd.DataFrame({ "Metric": 
                        [ 
                            "Number_of_Clusters", 
                            "Older_Clusters", 
                            "Newer_Clusters", 
                            "S_old_Size", 
                            "S_new_Size", 
                            "AUC_CV_Mean", 
                            "AUC_CV_Std", 
                            "AUC_Out_of_Fold", 
                            "Drift_Threshold", 
                            "Drift_Verdict", 
                            "Proceed", 
                            "Training_Time_Seconds" 
                        ], 
                        "Value": [ 
                            number_of_clusters, 
                            str(older_clusters), 
                            str(newer_clusters), 
                            len(S_old_final), 
                            len(S_new_final), 
                            round(auc_mean, 4), 
                            round(auc_std, 4), 
                            round(cv_auc_out_of_fold, 4), 
                            DRIFT_THRESHOLD, 
                            verdict, 
                            proceed, 
                            round(training_time, 4) 
                            ] 
                        }) 
summary.to_csv( "../results/tables/adversarial_validation_summary.csv", index=False ) 
print("FINAL ADVERSARIAL VALIDATION SUMMARY") 
print( summary.to_string(index=False) ) 
print( "\nAll adversarial-validation results saved successfully." )

# %%



