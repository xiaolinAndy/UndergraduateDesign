import pickle
import random

response_num = 20
evaluate_num = 60

twitter_context = open('./twitter/evaluate.responses.txt', 'r').readlines()
ubuntu_context = open('./ubuntu/evaluate.responses.txt', 'r').readlines()
tieba_context = open('./tieba/evaluate.responses.txt', 'r').readlines()

twitter_tfidf_response = open('./twitter/evaluate.tfidf.responses.txt', 'r').readlines()
ubuntu_tfidf_response = open('./ubuntu/evaluate.tfidf.responses.txt', 'r').readlines()
tieba_tfidf_response = open('./tieba/evaluate.tfidf.responses.txt', 'r').readlines()

twitter_dual_encoder_response = open('./twitter/evaluate.dual_encoder.responses.txt', 'r').readlines()
ubuntu_dual_encoder_response = open('./ubuntu/evaluate.dual_encoder.responses.txt', 'r').readlines()
tieba_dual_encoder_response = open('./tieba/evaluate.dual_encoder.responses.txt', 'r').readlines()

twitter_VHERD_response = open('./twitter/evaluate.VHERD.responses.txt', 'r').readlines()
ubuntu_VHERD_response = open('./ubuntu/evaluate.VHERD.responses.txt', 'r').readlines()
tieba_VHERD_response = open('./tieba/evaluate.VHERD.responses.txt', 'r').readlines()

tfidf_order = random.shuffle(range(500))
dual_encoder_order = random.shuffle(range(500))
VHERD_order = random.shuffle(range(500))
human_order = random.shuffle(range(500))

for i in range(1):
    #context for human responses:
    contexts = {'twitter': twitter_context[i*response_num:(i+1)*response_num],
                'ubuntu': ubuntu_context[i*response_num:(i+1)*response_num],
                'tieba': tieba_context[i*response_num:(i+1)*response_num]}
    evaluation_pairs = {}
    evaluation_pairs['twitter'] = {'tfidf'}
