from collections import defaultdict
import trueskill
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec


trueskill.setup(mu=25, sigma=8.333, beta=4.166, tau=0.08333, draw_probability=0.0)

df = pd.read_csv('races_cleaned.csv', sep=';')
race_took_place = df['status'] == 'finished'
df = df[race_took_place]

ratings = defaultdict(trueskill.Rating)

for index, row in df.iterrows():
    c = row['challenger']
    o = row['opponent']
    w = row['winner']
    if len({c, o, w}) != 2:
        raise ValueError(f'Illegal combination of challenger {c}, opponent {o}, and winner {w}.')
    winner, loser = (c, o) if c == w else (o, c)
    rating_pair = (ratings[winner], ratings[loser])
    ratings[winner], ratings[loser] = trueskill.rate_1vs1(*rating_pair)

leaderboard = sorted(ratings.items(), key=lambda item: trueskill.expose(item[1]), reverse=True)

counts_c = Counter(df['challenger'].value_counts())
counts_o = Counter(df['opponent'].value_counts())
counts = counts_c + counts_o

X, Y = [], []
for idx, skill in leaderboard:
    X.append(trueskill.expose(skill))
    Y.append(counts[idx])


fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)

ax_main.scatter(X, Y, marker='.')
ax_main.set(xlabel="conservative TrueSkill score", ylabel="games played")
plt.yscale('log')

ax_xDist.hist(X, bins=100, align='mid')
ax_xDist.set(ylabel='count')

ax_yDist.hist(Y, bins=10000, orientation='horizontal', align='mid')
ax_yDist.set(xlabel='count')

plt.show()

corr, _ = pearsonr(X, Y)
print(corr)
