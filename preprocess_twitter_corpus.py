import pickle
import random
import re

file_name = 'D:\Andy\Code\data\\twitter_big_corpus_cleaned_2.pkl'
pickle_file = open(file_name, "rb")
D = pickle.load(pickle_file)
print(D['count'])

output = open('D:\Andy\Code\data\\twitter_big_corpus_cleaned_3.pkl', 'w')
dialog = []
valid = 1

#co = re.compile(u'[\U00000100-\U0010ffff]')
for i, d in enumerate(D['data']):
    pair = []
    for j, s in enumerate(d):
        '''while '@@ ' in s:
            s = s.replace('@@ ', '')
        utterance = s.replace('@user', '<at>').replace('&lt;unk&gt;', '<unk>').replace('&lt;heart&gt;','<heart>').replace('&lt;number&gt;', '<number>').replace('  ', ' </s> ').replace('  ', ' ')
        # Make sure we end with </s> token
        utterance = utterance.replace('user', '<at>')
        utterance = utterance.replace('A:', '<first_speaker>')
        utterance = utterance.replace('B:', '<second_speaker>')
        utterance = utterance.replace('& lt', '<')
        utterance = utterance.replace('& gt', '>')
        utterance = utterance.replace('&lt;', '<')
        utterance = utterance.replace('&gt;', '>')
        utterance = utterance.replace('\'', ' \'')
        utterance = utterance.replace('"', ' " ')
        utterance = utterance.replace("'", " '")
        utterance = utterance.replace(";", " ")
        utterance = utterance.replace("`", " ")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace('.', ' . ')
        utterance = utterance.replace('!', ' ! ')
        utterance = utterance.replace('?', ' ? ')
        utterance = utterance.replace(',', ' , ')
        utterance = utterance.replace('~', '')
        utterance = utterance.replace('-', ' - ')
        utterance = utterance.replace('*', '')
        utterance = utterance.replace('(', ' ')
        utterance = utterance.replace(')', ' ')
        utterance = utterance.replace('[', ' ')
        utterance = utterance.replace(']', ' ')
        utterance = re.sub('[\s]+', ' ', utterance)
        utterance = utterance.replace('  ', ' ')
        utterance = utterance.replace('  ', ' ')
        s = utterance
        while '! ! ! !' in s:
            s = s.replace('! ! ! !', '! ! !')
        # s = utterance.replace('/', ' ')
        while len(s) > 0 and s[-1] == ' ':
            s = s[0:-1]
        if not s[-5:] == ' </s>':
            s = s + ' </s>
        s1 = co.sub(u'', s)
        pair.append(s)
        if (len(s1) != len(s)):
            valid = 0
        else:
            try:
                output.write(s + '\n')
            except:
                print(s)
                valid = 0
        if valid:
            dialog.append(pair)
        else:
            valid = 1'''
        output.write(s + '\n')
    if (i % 10000 == 0):
        print(i/10000)

'''data = {'name': 'twitter_big_corpus_cleaned_1',
        'count': D['count'],
        'max_length': D['max_length'],
        'min_length': D['min_length'],
        'data': dialog}
pickle.dump(data, new)'''
output.close()



