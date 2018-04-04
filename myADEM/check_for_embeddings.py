VHRED_FOLDER = '../vhred/'
# Import VHRED files.
import sys
sys.path.insert(0,VHRED_FOLDER)
from vhred_compute_dialogue_embeddings import compute_encodings
sys.path.remove(VHRED_FOLDER)
reload(sys)
sys.setdefaultencoding('utf8')
import argparse
from pretrain import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='default_config')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
    args = parse_args()
    config = eval(args.prototype)()
    c = [['今天', '天气', '真', '热']]
    r_gt = [['是', '啊', '我', '都', '出汗', '了']]
    r_model = [['对', '我', '现在', '直', '冒汗']]

    pretrainer = VHRED(config)
    c, r_gt, r_model = pretrainer._convert_text_to_bpe(c, r_gt, r_model)
