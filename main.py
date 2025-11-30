import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
filepath = "heart.csv"
df = pd.read_csv(filepath)

df = df.dropna()



#In the data set, Male is represented by 1, and Female by 0

# Target is already encoded as 0 and 1
y = df["target"]              
X = df.drop("target", axis=1)

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Train classifier
rf = RandomForestClassifier(n_estimators=20, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:\n", cm/cm.sum())

print("Percent of True Negatives:", tn/cm.sum())
print("Percent of True Positives:", tp/cm.sum())
print("\nPercent of Type I Error (False Positive):", fp/cm.sum())
print("Percent of Type II Error (False Negative):", fn/cm.sum())



#print(df.head(10))
#print(df.columns)
'''
print(df.shape)

plt.scatter(df['SamplingYear'], df['Diameter_inCentimeters'])
plt.show()

df=df.drop_duplicates()

#Drop rows with missing values
df=df.dropna()

df["SamplingYear"]=df["SamplingYear"].fillna("Unknown")
#OR fill missing values
#Can also do mean
df["Diameter_inCentimeters"]=df["Diameter_inCentimeters"].fillna(df["Diameter_inCentimeters"].median())

df["SamplingYear"]=pd.to_datetime(df["SamplingYear"], format="%Y")

#Standardize dates, Strings and round numberical data
#df["Date"]=pd.to_datetime(df["Date"], errors='coerce')
#df["Name"]=df["Name"].str.lower().str.strip()
#df["Score"]=df["Score"].round(2)

'''
