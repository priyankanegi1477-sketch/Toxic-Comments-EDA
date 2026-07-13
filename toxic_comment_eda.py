import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("train.csv.zip")
print("shape:")

print(df.shape)
print("info:")
print(df.info())
print("describe:")
print(df.describe())


"""
summary task 1 :
the information about dataset is described.

"""
print ("Missing values:")
print(df.isnull().sum())
print(df.isnull().mean())
"""
summary task 2 :
missing values checked
"""
print ("Duplicate analysis:\n")
print(df.duplicated().sum())
print(df.duplicated().mean())
"""
summary task 3:
"""
print("Label Distribution:")

print(df["toxic"].value_counts())
print(df["severe_toxic"].value_counts())
print(df["obscene"].value_counts())
print(df["threat"].value_counts())
print(df["insult"].value_counts())
print(df["identity_hate"].value_counts())
"""
summary task 4 :
each label is counted

"""

print("Percentage toxic:")
toxic_comments = round(df["toxic"].mean()*100, 2)
print(toxic_comments)
print("Percentage severe toxic:")
severe_toxic_comments = round(df["severe_toxic"].mean()*100, 2)
print(severe_toxic_comments)
print("Percentage obscene toxic:")
obscene_toxic_comments =round(df["obscene"].mean()*100, 2)
print(obscene_toxic_comments)
print("Percentage threat:")
threat_toxic_comments = round(df["threat"].mean()*100, 2)
print(threat_toxic_comments)
print("Percentage insult:")
insult_toxic_comments = round(df["insult"].mean()*100, 2)
print(insult_toxic_comments)
print("Percentage identity_hate:")
identity_hate_toxic_comments = round(df["identity_hate"].mean()*100, 2)
print(identity_hate_toxic_comments)

"""
summary task 5 :
each category is described in percentage 

"""
df["comment_length"] = df["comment_text"].str.len()  # comment_text already exists in csv just see
print(df["comment_length"].describe())


avg_toxic_len = round(df.groupby("toxic")["comment_length"].mean(), 0)
avg_severe_toxic_len = round(df.groupby("severe_toxic")["comment_length"].mean(), 0)
avg_insult_len= round(df.groupby("insult")["comment_length"].mean(),0)
avg_identity_len= round(df.groupby("identity_hate")["comment_length"].mean(), 0)
avg_threat_len = round(df.groupby("threat")["comment_length"].mean(),0)
avg_obscene_len= round(df.groupby("obscene")["comment_length"].mean(),0)

print(avg_toxic_len)
print(avg_severe_toxic_len)
print(avg_insult_len)
print(avg_identity_len)
print(avg_threat_len)
print(avg_obscene_len)
"""
summary task 7:
avg length of comment in each category
"""
labels = ["toxic", "insult", "identity_hate", "threat", "obscene", "severe_toxic"]
df["toxicity_count"] = df[labels].sum(axis=1)
print("new toxicity count:")

print(df["toxicity_count"].value_counts())

"""
summary task 8:
31 comments have all 6 categories toxicity but a lot of them are clean comments . 
"""
print("toxic comment sorted based on toxicity:")
print(df.sort_values(by = "toxicity_count", ascending = False )[["comment_text", "toxicity_count"]].head(15))
print(len(df[df["toxicity_count"]== 6]))
"""
summary task 9:
sorted top 15 toxic count

"""
print("correlation in top comments :")
print(df[labels].corr())

"""
summary task 10:
the numbers close to 1 means highly related and vice versa 
"""
"""print(df["threat"].mean()*100)
print(df["toxic"].mean()*100)"""

y = [df["toxic"].sum(),df["severe_toxic"].sum(),
 df["insult"].sum(),df["obscene"].sum(),
 df["identity_hate"].sum(),df["threat"].sum(),]

x= ["toxic","severe toxic" , "insult","obscene","identity_hate","threat"]

plt.bar(x,y)
plt.xlabel("Categories")
plt.ylabel("Count")
plt.title("Toxic comments")
plt.show()
