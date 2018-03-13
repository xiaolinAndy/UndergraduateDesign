# coding=gbk
f = open('../Data/tieba_comments.txt', 'rb')
dialogues = []
D = f.readlines()
for i, line in enumerate(D):
    if i < 185075:
        continue
    if i % 2 == 0:
        if i+2 < len(D) and D[i] == D[i+2]:
           dialogues.append([])
        elif i+2 >= len(D):
            dialogues.append([])
        else:
            count = 1
            while (i+2*count+2 < len(D) and D[i+2*count] != D[i+2*count+2] and count < 20):
                count += 1
            if count > 8:
                print(i)
                break
    else:
        continue


