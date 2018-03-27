This repository has the code and parameters used for the ADEM model in:

**Towards an Automatic Turing Test: Learning to Evaluate Dialogue Responses**  
Ryan Lowe, Michael Noseworthy, Iulian V. Serban, Nicolas Angelard-Gontier, Yoshua Bengio, and Joelle Pineau

Due to the ethics policy for this project, we cannot release the collected human data at this time.
However, we do provide the weights/parameters for a trained model and the code to train ADEM with new data.

ADEM uses the VHRED model. A modified version of the code is included in this repo. The original repo and paper can be found at:  
https://github.com/julianser/hed-dlg-truncated  
https://arxiv.org/abs/1605.06069

You will need to download the weights for the pretrained VHRED model before running the code. Once downloaded from the following link, place all the files in the `./vhred` folder.  
https://drive.google.com/file/d/0B-nb1w_dNuMLY0Fad3N1YU9ZOU0/view?usp=sharing

An example of running ADEM can be found in `interactive.py`:  
`THEANO_FLAGS='device=gpu0,floatX=float32' python interactive.py`

Things need todo:
--1. change the directory of dict in config['vhred_dict']
--2. change the directory of vhred_prefix in config['vhred_prefix']
--3. kill all bpe
--4. focus on the format of dataset: { 'c': context, 'r_gt': true response, 'r_models': {'hred': (model_response, score), ... }}
--5. rewrite _strs_to_idxs and _idxs_to_strs without bpe
(question)--6. when using the output of encoder try to use only h instead of [h, hs]
--8. way to compute last_token_position should be changed
--9. Create arrays to store the data. The middle dimension represents: # 0: context, 1: gt_response, 2: model_response
(no sample at the present)10. reconsider about sampling, need to do data analysis first
--11. when _compute_init_values, 'i' maybe wrong?
(question)--12. train_x becomes shared?
--13. M,N 5000 parameters, overfitting? Solved, data is enough complex
14. try to delete the <end> in the front of each sentence and see what will happen
15. how to train alpha, beta, no need to train? maybe use sigmoid instead?

