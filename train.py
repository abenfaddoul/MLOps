# Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from skops.io import dump, load, get_untrusted_types
import datetime

# Loading the Dataset

drug_df = pd.read_csv("./Data/drug.csv")
drug_df = drug_df.sample(frac=1)


# Train Test Split

X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

''' Machine Learning Pipelines
We will build a processing pipeline using ColumnTransformer, which will convert categorical values into numbers, fill in missing values, and scale the numerical columns.
After that, we'll create a training pipeline that will take the transformed data and train a random forest classifier.
Finally, we'll train the model.
By using pipelines, we can ensure reproducibility, modularity, and clarity in our code. '''

cat_col = [1,2,3]
num_col = [0,4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)
pipe.fit(X_train, y_train)


# Model Evaluation

predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))


## Confusion Matrix Plot

predictions = pipe.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("./Results/model_results.png", dpi=120)

## Write metrics to file
with open("./Results/metrics.txt", "a") as outfile:
    time_stamp = datetime.datetime.now()
    time = time_stamp.strftime("%d-%m-%y %H:%M:%S")
    outfile.write(f"\n{time} Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

## Saving the model file
dump(pipe, "./Model/drug_pipeline.skops")