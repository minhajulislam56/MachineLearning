import pandas as pd
import re   # regular expression
# df = pd.read_csv('pokemon_data.csv')
#df = pd.read_csv('pokemon_data.txt', delimiter='\t')
df = pd.read_csv('new_pokemon.csv')

new_df = pd.DataFrame(columns=df.columns)
print(new_df)
for idf in pd.read_csv('new_pokemon.csv', chunksize=5):
    counting = idf.groupby(['Type 1']).count()
    new_df = pd.concat([new_df, counting])

print(new_df)

# print(df.groupby(['Generation']).mean().sort_values('Sum', ascending=False))    # Groupby
# df['count'] = 1
# df.groupby(['Type 1', 'Type 2']).count()['count']
# df = df.groupby(['Type 1', 'Type 2']).count().sort_values('count', ascending=False)
# print(df)

# print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)])
# print(df.loc[df['Name'].str.contains('^pi[a-z]*', flags=re.IGNORECASE, regex=True)])
# df.loc[df['Type 1'] == 'Grass', ['Speed', 'Generation']] = [0, 2]
# print(df)
# without_Mega = df.loc[~df['Name'].str.contains('Mega')]
# print(without_Mega)
# filter_df = df.loc[(df['Type 1'] == 'Grass') | (df['Defense'] > 100)]
# filter_df.reset_index(drop=True, inplace=True)
# print(filter_df)
# df['Sum'] = df['Attack'] + df['Defense']
# df['Sum'] = df.iloc[:, 4:6].sum(axis=1)
# col = list(df.columns)
# df = df[col[0:4] + [col[-1]] + col[4:12]]
# print(df.head(4))
# df.to_csv('new_pokemon.csv', index=False)
# df.to_csv('new_pokemon.txt', index=False, sep='\t')
# print(df.iloc[:, 4].sum(axis=0))
# df = df.drop(columns='Sum')
# print(df.head(4))
# print(df.sort_values(['Defense', 'Speed'], ascending=[False, True])[['Name', 'Defense', 'Speed']])
# print (df.loc[(df['Attack'] >= 60) & (df['Type 1'] == 'Grass')][['Name', 'Defense', 'Type 1']])
# for index, row in df.iterrows():
#     print(row[['Name', 'HP']])
# print(df.iloc[9,1])
# print(df.iloc[0:2])
# print(df[['Name', 'HP']][0:5])
# print(df.tail(2))