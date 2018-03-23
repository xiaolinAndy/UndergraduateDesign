import pickle
import random

#response_num = 30
evaluate_num = 200

#twitter_context = open('./twitter/evaluate.contexts.txt', 'r').readlines()
#ubuntu_context = open('./ubuntu/evaluate.contexts.txt', 'r').readlines()
tieba_context = open('./tieba/new.tieba.contexts.txt', 'r', encoding='utf-8').readlines()

#twitter_tfidf_response = open('./twitter/evaluate.tfidf.responses.txt', 'r').readlines()
#ubuntu_tfidf_response = open('./ubuntu/evaluate.tfidf.responses.txt', 'r').readlines()
tieba_tfidf_response = open('./tieba/new.tfidf.responses.txt', 'r', encoding='utf-8').readlines()

#twitter_dual_encoder_response = open('./twitter/evaluate.dual_encoder.responses.txt', 'r').readlines()
#ubuntu_dual_encoder_response = open('./ubuntu/evaluate.dual_encoder.responses.txt', 'r').readlines()
tieba_dual_encoder_response = open('./tieba/new.dual_encoder.responses.txt', 'r', encoding='utf-8').readlines()

#twitter_VHERD_response = open('./twitter/evaluate.VHERD.responses.txt', 'r').readlines()
#ubuntu_VHERD_response = open('./ubuntu/evaluate.VHERD.responses.txt', 'r').readlines()
tieba_VHERD_response = open('./tieba/new.VHERD.responses.txt', 'r', encoding='utf-8').readlines()

tieba_human_response = open('./tieba/new.human.responses.txt', 'r').readlines()

response_order = list(range(2000))
human_order = list(range(500))
random.shuffle(response_order)
random.shuffle(human_order)
#twitter_response = twitter_tfidf_response + twitter_dual_encoder_response + twitter_VHERD_response
#ubuntu_response = ubuntu_tfidf_response + ubuntu_dual_encoder_response + ubuntu_VHERD_response
tieba_response = tieba_tfidf_response + tieba_dual_encoder_response + tieba_VHERD_response + tieba_human_response

for i in range(10):
    #context for human responses:
    '''ind = human_order[i*response_num:(i+1)*response_num]
    contexts = {#'twitter': [twitter_context[ind[i]] for i in range(response_num)],
                #'ubuntu': [ubuntu_context[ind[i]] for i in range(response_num)],
                'tieba': [tieba_context[ind[i]] for i in range(response_num)]}'''
    ind = response_order[i*evaluate_num:(i+1)*evaluate_num]
    twitter_p = []
    #ubuntu_p = []
    tieba_p = []
    for j in ind:
        #twitter_p.append([twitter_context[j % 500], twitter_response[j]])
        #ubuntu_p.append([ubuntu_context[j % 500], ubuntu_response[j]])
        tieba_p.append([tieba_context[j % 500], tieba_response[j]])
    evaluation_pairs = {#'twitter': twitter_p,
                        #'ubuntu': ubuntu_p,
                        'tieba': tieba_p}
    '''task1 = open('./data/task1_' + str(i+1), 'wb')
    pickle.dump(contexts, task1)'''
    task2 = open('./data/task_' + str(i + 1), 'wb')
    pickle.dump(evaluation_pairs, task2)
    index = open('./data/index', 'wb')
    pickle.dump({#'human_order': human_order,
                 'response_order': response_order}, index)