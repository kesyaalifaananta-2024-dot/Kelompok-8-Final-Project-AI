# In[1]:


# fetch dataset from same folder
import pandas as pd

train = pd.read_csv(
    "train.csv",
    low_memory=False,
    dtype={"StateHoliday": str} # force pandas to read StateHoliday as string
)

store = pd.read_csv(
    "store.csv"
)

# merge dataset
df = train.merge(
    store,
    on="Store",
    how="left"
)

# check
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.dtypes)


# In[2]:


# data cleaning
# replace missing values (int and float) with median
num_cols = [
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear"
]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# replace PromoInterval (str) with None
df["PromoInterval"] = df["PromoInterval"].fillna("None")

# remove when stores are closed
df = df[df["Open"] == 1]

# remove 0 sales
df = df[df["Sales"] > 0]

# convert date
df["Date"] = pd.to_datetime(df["Date"])

# sort chronologically
df = df.sort_values(["Store", "Date"])

# verify
print(df.duplicated().sum())
print(df.isnull().sum())

# In[5]:


# prepare features and target
features = [
    "Customers",
    "Promo",
    "Promo2",
    "SchoolHoliday",
    "CompetitionDistance",
    "lag1",
    "lag7",
    "lag14",
    "lag30",
    "rolling7",
    "rolling14",
    "rolling30",
    "Month",
    "Week",
    "DayOfWeek",
]

import numpy as np

df["Sales"] = np.log1p(df["Sales"])


# In[6]:


X = df[features]
y = df["Sales"]

from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)

y_scaled = y_scaler.fit_transform(
    y.values.reshape(-1, 1)
)

train_size = int(len(X_scaled) * 0.70)
val_size = int(len(X_scaled) * 0.15)

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]

X_val = X_scaled[
    train_size:train_size + val_size
]

y_val = y_scaled[
    train_size:train_size + val_size
]

X_test = X_scaled[
    train_size + val_size:
]

y_test = y_scaled[
    train_size + val_size:
]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# In[7]:


import numpy as np

X_train_val = np.vstack((X_train, X_val))
y_train_val = np.vstack((y_train, y_val))
