import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('combined.csv')
df = df.set_index(df['epoch'])
df = df.drop(columns=['epoch'])
print(df.head())
df.plot()
plt.ylabel('Accuracy %')
plt.title('Accuracy Growth by Epoch and CNN Depth')
plt.show()

