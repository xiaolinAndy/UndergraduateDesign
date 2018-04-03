from nltk.translate.bleu_score import sentence_bleu
import argparse
from preprocess import load_data
from scipy.stats import pearsonr, spearmanr
from experiments import *
from pythonrouge.pythonrouge import Pythonrouge
import numpy as np
import cPickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='default_config')
    args = parser.parse_args()
    return args


def _correlation(output, score):
    return [spearmanr(output, score), pearsonr(output, score)]

def calculate_sentence_correlation(index_start, index_end, config):
    data = cPickle.load(open(config['exp_folder'] + '/dataset.pkl', 'rb'))
    scores_1, scores_2, scores_3, scores_4 = [], [], [], []
    real_scores, real_scores_1, real_scores_2 = [], [], []
    for entry in data[index_start:index_end]:
        r_gt = entry['r_gt']
        r_models = entry['r_models']
        for key in r_models.keys():
            scores_1.append(sentence_bleu([r_gt], r_models[key][0], weights=(1, 0, 0, 0)))
            scores_2.append(sentence_bleu([r_gt], r_models[key][0], weights=(0.5, 0.5, 0, 0)))
            scores_3.append(sentence_bleu([r_gt], r_models[key][0], weights=(0.33, 0.33, 0.33, 0)))
            scores_4.append(sentence_bleu([r_gt], r_models[key][0], weights=(0.25, 0.25, 0.25, 0.25)))
            real_scores.append(r_models[key][1][0])
            real_scores_1.append(r_models[key][1][1])
            real_scores_2.append(r_models[key][1][2])
    #print len(scores_1), len(real_scores)
    cor_1 = _correlation(scores_1, real_scores)
    cor_2 = _correlation(scores_2, real_scores)
    cor_3 = _correlation(scores_3, real_scores)
    cor_4 = _correlation(scores_4, real_scores)
    cor_h = _correlation(real_scores_1, real_scores_2)
    print cor_1, '\n', cor_2, '\n', cor_3, '\n', cor_4, '\n', cor_h

def calculate_model_correlation(index_start, index_end, config, score=None, order=None):
    data = cPickle.load(open(config['exp_folder'] + '/dataset.pkl', 'rb'))
    if order == None:
        scores_1, scores_2, scores_3, scores_4 = {'tfidf':0,'de':0,'vhred':0,'human':0}, {'tfidf':0,'de':0,'vhred':0,'human':0}, {'tfidf':0,'de':0,'vhred':0,'human':0}, {'tfidf':0,'de':0,'vhred':0,'human':0}
        real_scores, real_scores_1, real_scores_2 = {'tfidf':0,'de':0,'vhred':0,'human':0}, {'tfidf':0,'de':0,'vhred':0,'human':0}, {'tfidf':0,'de':0,'vhred':0,'human':0}
        for entry in data[index_start:index_end]:
            r_gt = entry['r_gt']
            r_models = entry['r_models']
            for key in r_models.keys():
                scores_1[key] += sentence_bleu([r_gt], r_models[key][0], weights=(1, 0, 0, 0))
                scores_2[key] += sentence_bleu([r_gt], r_models[key][0], weights=(0.5, 0.5, 0, 0))
                scores_3[key] += sentence_bleu([r_gt], r_models[key][0], weights=(0.33, 0.33, 0.33, 0))
                scores_4[key] += sentence_bleu([r_gt], r_models[key][0], weights=(0.25, 0.25, 0.25, 0.25))
                real_scores[key] += r_models[key][1][0]
                real_scores_1[key] += r_models[key][1][1]
                real_scores_2[key] += r_models[key][1][2]
        scores_1 = list(scores_1.values())
        scores_2 = list(scores_2.values())
        scores_3 = list(scores_3.values())
        scores_4 = list(scores_4.values())
        real_scores = list(real_scores.values())
        real_scores_1 = list(real_scores_1.values())
        real_scores_2 = list(real_scores_2.values())
        cor_1 = _correlation(scores_1, real_scores)
        cor_2 = _correlation(scores_2, real_scores)
        cor_3 = _correlation(scores_3, real_scores)
        cor_4 = _correlation(scores_4, real_scores)
        cor_h = _correlation(real_scores_1, real_scores_2)
        #print scores_1, scores_2, scores_3, scores_4, real_scores
        print cor_1, '\n', cor_2, '\n', cor_3, '\n', cor_4, '\n', cor_h
    else:
        model_scores = {'tfidf': 0, 'de': 0, 'vhred': 0, 'human': 0}
        real_scores = {'tfidf': 0, 'de': 0, 'vhred': 0, 'human': 0}
        for entry in data[index_start:index_end]:
            r_models = entry['r_models']
            for key in r_models.keys():
                real_scores[key] += r_models[key][1][0]
        for i,key in enumerate(order):
            model_scores[key] = np.mean(score[i::4])
        cor_1 = _correlation(list(model_scores.values()), list(real_scores.values()))
        print cor_1, '\n'


if __name__ == "__main__":
    args = parse_args()
    config = eval(args.prototype)()
    print 'Beginning...'
    # This will load our training data.
    data = load_data(config)
    calculate_sentence_correlation(0, len(data), config)
    calculate_model_correlation(0, len(data), config)
