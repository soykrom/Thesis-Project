import json
import pandas

# Import data 
df_mine = pandas.read_csv("../volta3.csv", sep = '|')

df1_mine = df_mine.iloc[:,1:]   # select just the timestamp and value columns
df2_mine = df1_mine.iloc[1:,:]  # eliminate first row
df2_mine = df2_mine.iloc[:-1,:] # eliminate last row
df2_mine.columns = ['timestamp', 'value']
df3_mine = df2_mine.sort_values(by=['timestamp'])
df3_mine.iloc[:,1] = df3_mine.iloc[:,1].apply(str.strip)

rows = []
for i in range (1, df3_mine.shape[0]):
    rows.append(df3_mine.at[i, 'value'])

with open('data.json', 'w') as f:
    f.write(str(rows).replace("\'", "") + '\n')
exit()