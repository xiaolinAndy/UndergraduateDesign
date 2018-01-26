import pickle

file_name = 'D:\Andy\Data\\twitter_small_corpus.txt'
dialog = []
count_line = 0
max_length = 0
min_length = 10000
total_length = 0
with open(file_name, 'r', encoding='utf-8') as f:
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line1 :
            break
        count_line += 2
        dialog.append([line1, line2])
        words1 = line1.split()
        words2 = line2.split()
        total_length += len(words1)  + len(words2)
        max_length = max(max_length, len(words1), len(words2))
        min_length = min(min_length, len(words1), len(words2))
    avg_length = float(total_length) / count_line
    print(count_line, max_length, min_length, avg_length)
    data = {'name': 'twitter_short_corpus',
            'count': count_line/2,
            'max_length': max_length,
            'min_length': min_length,
            'data': dialog}
    output = open('D:\Andy\Code\data\\twitter_short_corpus.pkl', 'wb')
    pickle.dump(data, output)
    output.close()






