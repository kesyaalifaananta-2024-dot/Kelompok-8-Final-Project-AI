# In[4]:


# features
# date
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Quarter"] = df["Date"].dt.quarter

# lag (sales history)
df["lag1"] = (df.groupby("Store")["Sales"].shift(1)) # 1 day ago
df["lag7"] = (df.groupby("Store")["Sales"].shift(7)) # 7 days ago
df["lag14"] = (df.groupby("Store")["Sales"].shift(14)) # 14 days ago
df["lag30"] = (df.groupby("Store")["Sales"].shift(30)) # 30 days ago

# rolling (average sales)
df["rolling7"] = (
    df.groupby("Store")["Sales"]
        .transform(lambda x: x.rolling(7).mean()) # 7 days
)
df["rolling14"] = (
    df.groupby("Store")["Sales"]
        .transform(lambda x: x.rolling(14).mean()) # 14 days
)
df["rolling30"] = (
    df.groupby("Store")["Sales"]
        .transform(lambda x: x.rolling(30).mean()) # 30 days
)

df["rolling7_std"] = (
    df.groupby("Store")["Sales"]
      .transform(lambda x: x.rolling(7).std()) # sales variability in 7 days
)

df["rolling30_std"] = (
    df.groupby("Store")["Sales"]
      .transform(lambda x: x.rolling(30).std()) # sales variability in 30 days
)

# competition (how long competitors have been open)
df["CompetitionOpenYears"] = (df["Year"] - df["CompetitionOpenSinceYear"])

# remove missing values from lag and rolling
df = df.dropna()
