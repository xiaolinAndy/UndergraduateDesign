import pickle

index = pickle.load(open('./data/index', 'rb'))
tieba_s = open('./data/statistics_1.txt', 'r').readline()
ord = index['response_order'][:200]
tfidf_score = {1:0,2:0,3:0,4:0,5:0}
dual_encoder_score = {1:0,2:0,3:0,4:0,5:0}
VHRED_score = {1:0,2:0,3:0,4:0,5:0}
human_score = {1:0,2:0,3:0,4:0,5:0}
scores = tieba_s.split()
print(len(scores))
for i, ind in enumerate(ord):
    if int(ind/500) == 0:
        tfidf_score[int(scores[i])] += 1
    elif int(ind/500) == 1:
        dual_encoder_score[int(scores[i])] += 1
    elif int(ind/500) == 2:
        VHRED_score[int(scores[i])] += 1
    else:
        human_score[int(scores[i])] += 1
tmp = 0
count = 0
for p in tfidf_score:
    tmp += tfidf_score[p] * p
    count += tfidf_score[p]
    print(p, ': ', tfidf_score[p])
print('tfidf: ', tmp / count)
count = 0
tmp = 0
for p in dual_encoder_score:
    count += dual_encoder_score[p]
    tmp += dual_encoder_score[p] * p
    print(p, ': ', dual_encoder_score[p])
print('dual_encoder: ', tmp / count)
count = 0
tmp = 0
for p in VHRED_score:
    count += VHRED_score[p]
    tmp += VHRED_score[p] * p
    print(p, ': ', VHRED_score[p])
print('VHRED_score: ', tmp / count)
count = 0
tmp = 0
for p in human_score:
    count += human_score[p]
    tmp += human_score[p] * p
    print(p, ': ', human_score[p])
print('human_score: ', tmp / count)