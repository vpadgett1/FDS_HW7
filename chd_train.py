import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler

# Read the training data
print("Reading input...")
## Your code here
df = pd.read_csv("train.csv", index_col=False)
#Divides it into two numpy arrays: Y is TenYearCHD, X is the other columns.
Y = df["TenYearCHD"]
X = df.drop("TenYearCHD", axis=1)

print("Scaling...")
## Your code here
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Fitting...")
## Your code here
model = LogisticRegression()
model.fit(X, Y)
train_accuracy = model.score(X, Y)
# Get the accuracy on the training data
## Your code here
print(f"Training accuracy = {train_accuracy}")

# Write out the scaler and logisticregression objects into a pickle file
pickle_path = "classifier.pkl"
print(f"Writing scaling and logistic regression model to {pickle_path}...")
## Your code here
with open(pickle_path, "wb") as f:
    pickle.dump((scaler, model), f)