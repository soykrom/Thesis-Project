
import pandas

# Import data 
df_mine = pandas.read_csv("5041651648624125_volta3.csv",sep = '|')

df1_mine = df_mine.iloc[:,1:]   # select just the timestamp and value columns
df2_mine = df1_mine.iloc[1:,:]  # eliminate first row
df2_mine = df2_mine.iloc[:-1,:] # eliminate last row
df2_mine.columns = ['timestamp','value']
df3_mine = df2_mine.sort_values(by=['timestamp'])
df3_mine.iloc[:,1] = df3_mine.iloc[:,1].apply(str.strip)

print(df3_mine)

exit()

