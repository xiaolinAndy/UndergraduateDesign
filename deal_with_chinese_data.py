f = open('../tieba/temp_comments_clean.txt', 'r')
dialogues = []
D = f.readlines()
for i, line in enumerate(D):
    #if i < 185027:
        #continue
    if i % 2 == 0:
        if i+2 < len(D) and D[i] == D[i+2]:
           dialogues.append([])
        elif i+2 >= len(D):
            dialogues.append([])
        else:
            count = 1
            while (i+2*count+2 < len(D) and D[i+2*count] != D[i+2*count+2] and count < 20):
                count += 1
            if count > 9:
                print(i)
                break
    else:
        continue
'''f = open('../tieba/temp_comments_clean_2.txt', 'r')
ff = open('../tieba/temp_comments_clean_3.txt', 'w')
dialogues = []
D = f.readlines()
j = 0
flag = 0
for i, line in enumerate(D):
    if i % 2 == flag:
        if i+2 < len(D) and D[i] == D[i+2]:
            ff.write(line)
        elif i+2 >= len(D):
            ff.write(line)
        else:
            count = 1
            while (i+2*count+2 < len(D) and D[i+2*count] != D[i+2*count+2] and count < 20):
                count += 1
            if count > 8:
                print(i)
                #break
                flag = 1 - flag
            else:
                ff.write(line)
    else:
        ff.write(line)'''



