import pickle
from scipy.stats import pearsonr, spearmanr

index = pickle.load(open('./data/index', 'rb'))
tieba_s = open('./data/statistics.txt', 'r').readlines()
ord = index['response_order']
score_1, score_2, avg = [], [], []
tfidf_score = {1:0,1.5:0,2:0,2.5:0,3:0,3.5:0,4:0,4.5:0,5:0}
dual_encoder_score = {1:0,1.5:0,2:0,2.5:0,3:0,3.5:0,4:0,4.5:0,5:0}
VHRED_score = {1:0,1.5:0,2:0,2.5:0,3:0,3.5:0,4:0,4.5:0,5:0}
human_score = {1:0,1.5:0,2:0,2.5:0,3:0,3.5:0,4:0,4.5:0,5:0}
for i,line in enumerate(tieba_s):
    if i % 2 == 0:
        score_1 += [int(s) for s in line.split()]
    else:
        score_2 += [int(s) for s in line.split()]
for i in range(len(ord)):
    avg.append((score_1[i]+score_2[i])/2.0)
for i, ind in enumerate(ord):
    if int(ind/387) == 0:
        tfidf_score[avg[i]] += 1
    elif int(ind/387) == 1:
        dual_encoder_score[avg[i]] += 1
    elif int(ind/387) == 2:
        VHRED_score[avg[i]] += 1
    else:
        human_score[avg[i]] += 1
tmp = 0
count = 0
for p in tfidf_score:
    tmp += tfidf_score[p] * p
    count += tfidf_score[p]
    print(p, ': ', tfidf_score[p])
print('tfidf: ', tmp / count, count)
count = 0
tmp = 0
for p in dual_encoder_score:
    count += dual_encoder_score[p]
    tmp += dual_encoder_score[p] * p
    print(p, ': ', dual_encoder_score[p])
print('dual_encoder: ', tmp / count, count)
count = 0
tmp = 0
for p in VHRED_score:
    count += VHRED_score[p]
    tmp += VHRED_score[p] * p
    print(p, ': ', VHRED_score[p])
print('VHRED_score: ', tmp / count, count)
count = 0
tmp = 0
for p in human_score:
    count += human_score[p]
    tmp += human_score[p] * p
    print(p, ': ', human_score[p])
print('human_score: ', tmp / count, count)

statistics = {'score_1': score_1,
              'score_2': score_2,
              'avg': avg,
              'tfidf': tfidf_score,
              'dual_encoder': dual_encoder_score,
              'VHRED': VHRED_score,
              'human': human_score}
f = open('./data/statistics.pkl', 'wb')
pickle.dump(statistics, f, protocol=2)

print(pearsonr(score_1, score_2), spearmanr(score_1, score_2))
t1, d1, v1 ,h1, t2, d2, v2, h2 = 0,0,0,0,0,0,0,0
for i,ind in enumerate(ord):
    if int(ind / 387) == 0:
        t1 += score_1[i]
        t2 += score_2[i]
    elif int(ind / 387) == 1:
        d1 += score_1[i]
        d2 += score_2[i]
    elif int(ind / 387) == 2:
        v1 += score_1[i]
        v2 += score_2[i]
    else:
        h1 += score_1[i]
        h2 += score_2[i]
s1 = [t1/387.0, d1/387.0, v1/387.0, h1/387.0]
s2 = [t2/387.0, d2/387.0, v2/387.0, h2/387.0]
print(pearsonr(s1, s2), spearmanr(s1, s2))