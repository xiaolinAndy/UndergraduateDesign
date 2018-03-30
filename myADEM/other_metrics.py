from nltk.translate.bleu_score import sentence_bleu
import argparse
from preprocess import load_data
from scipy.stats import pearsonr, spearmanr
from experiments import *
from pythonrouge.pythonrouge import Pythonrouge
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


if __name__ == "__main__":
    args = parse_args()
    config = eval(args.prototype)()
    print 'Beginning...'
    # This will load our training data.
    data = load_data(config)
    scores_1, scores_2, scores_3, scores_4 = [], [], [], []
    rouge_score = []
    real_scores = []
    for entry in data:
        r_gt = entry['r_gt']
        r_models = entry['r_models']
        for key in r_models.keys():
            scores_1.append(sentence_bleu([r_gt], r_models[key][0], weights=(1, 0, 0, 0)))
            scores_2.append(sentence_bleu([r_gt], r_models[key][0], weights=(0.5, 0.5, 0, 0)))
            scores_3.append(sentence_bleu([r_gt], r_models[key][0], weights=(0.33, 0.33, 0.33, 0)))
            scores_4.append(sentence_bleu([r_gt], r_models[key][0], weights=(0.25, 0.25, 0.25, 0.25)))
            summary = [[" Tokyo is the one of the biggest city in the world."]]
            reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=summary, reference=reference,
                                n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                                recall_only=True, stemming=True, stopwords=True,
                                word_level=True, length_limit=True, length=50,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            #print rouge.calc_score()
            #rouge_score.append(rouge.calc_score()[2])
            real_scores.append(r_models[key][1])
    cor_1 = _correlation(scores_1, real_scores)
    cor_2 = _correlation(scores_2, real_scores)
    cor_3 = _correlation(scores_3, real_scores)
    cor_4 = _correlation(scores_4, real_scores)
    #cor_rouge = _correlation(rouge_score, real_scores)
    print cor_1, '\n', cor_2, '\n', cor_3, '\n', cor_4
