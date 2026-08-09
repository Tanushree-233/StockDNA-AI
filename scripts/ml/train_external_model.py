import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


# ==================================================
# LOAD DATA
# ==================================================

X_train = joblib.load("models/X_external_train.pkl")
X_test = joblib.load("models/X_external_test.pkl")

y_train = joblib.load("models/y_external_train.pkl")
y_test = joblib.load("models/y_external_test.pkl")


print("=" * 60)
print("TRAINING EXTERNAL XGBOOST MODEL")
print("=" * 60)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ==================================================
# CLASS WEIGHTS
# ==================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


print("\nClass distribution:")
print(y_train.value_counts().sort_index())


# ==================================================
# MODEL
# ==================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=3,
    gamma=0.1,

    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)


# ==================================================
# TRAIN
# ==================================================

print("\nTraining...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


print("\nTraining complete!")


# ==================================================
# SAVE
# ==================================================

joblib.dump(
    model,
    "models/external_xgboost.pkl"
)

print("\nSaved:")
print("models/external_xgboost.pkl")