"""
This file runs the full automatic evaluation suite for any set of predictions on the TellMeWhy dataset
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from rouge_metric import PyRouge
import sacrebleu
import random
import time
from bert_score import BERTScorer

np.random.seed(1234)
random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-output-file', type=str, help='Predictions file path')
    parser.add_argument('--log-file', type=str, required=True, help='Log filepath to write information to')
    parser.add_argument('--temp-dir', type=str, required=True, help='Temporary directory path prefix for BLEURT to use')
    args, _ = parser.parse_known_args()
    return args

def create_multi_reference_dictionary_for_gold_sentences(df):

    """
    This function reads the entire test file and creates a dictionary.
    A combination of context and question are keys and a list of human answers for the same question is the value.
    This is important for calculating the ROUGE metric
    """

    source_multi_ref_dict = {}
    for idx, row in df.iterrows():
        try:
            source_multi_ref_dict[row['meta']].append(row['gold_answer'])
        except:
            source_multi_ref_dict[row['meta']] = [row['gold_answer']]
    return source_multi_ref_dict

def create_inputs_for_rouge(df, source_multi_ref_dict):

    """
    This function creates hypotheses and references array that is used to calculate ROUGE.
    Reference for ROUGE code: https://pypi.org/project/rouge-metric/
    """

    hypotheses, references = [], []
    for idx, row in df.iterrows():
        hypotheses.append(row['predicted_answer'])
        references.append(source_multi_ref_dict[row['meta']])
    assert len(hypotheses) == len(hypotheses)
    return references, hypotheses

def rouge(hypotheses, references):

    """
    This function performs the requisite ROUGE metric calculation.
    """

    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    rouge_scores = rouge.evaluate(hypotheses, references)
    scores = rouge_scores['rouge-l']
    log_str = 'ROUGE-L score'
    print(log_str)
    logging.info(log_str)
    for k, v in scores.items():
        score_types = {'r': 'recall', 'p': 'precision', 'f': 'F-score'}
        log_str = f'\t {score_types[k]} = {round(v, 2)}'
        print(log_str)
        logging.info(log_str)

def sentence_level_multi_bertscore(df, scorer):
    metas, references, system_outputs = [], [], []
    grouped_df = df.groupby(by=['meta'])
    for group_name, group in grouped_df:
        metas.append(group_name)
        system_outputs.append(group.iloc[0]['predicted_answer'])
        record_references = []
        selected_choices = [0, 1, 2]
        for choice in selected_choices:
            record_references.append(group.iloc[choice]['gold_answer'])
        references.append(record_references)
    P, R, F1 = scorer.score(system_outputs, references, verbose=False)
    return F1.tolist(), metas, F1.mean().item()

def sentence_level_single_reference_bleu_scores(system_outputs, references):

    """
    Calculate sentence level BLEU scores using sacrebleu.
    In essence, this just calls bleu() for each sentence pair in the corpus one by one.
    """

    bleu_scores = []
    for ref, sys_out in zip(references, system_outputs):
        # arguments to sacrebleu need to be lists
        score = bleu([[ref]], [sys_out])
        bleu_scores.append(score)
    return np.array(bleu_scores)

def bleu(references, system_outputs):

    """
    This function calculates BLEU score for references and predictions using sacrebleu
    """

    bleu = sacrebleu.corpus_bleu(system_outputs, references)

    return round(bleu.score, 2)

def remove_garbage_newlines(text):

    """
    Replace all instances of newline characters so that one record only uses one line in a file
    """

    new_text = text.replace('\n', '')
    return new_text

def single_reference_bleurt(df, temp_dir):

    """
    Create temporary files to store references and system outputs
    Run shell command for BLEURT and write scores to file
    Read scores from file and return final score
    """

    df['prep_gold_answer'] = df['gold_answer'].apply(remove_garbage_newlines)
    df['prep_predicted_answer'] = df['predicted_answer'].apply(remove_garbage_newlines)

    system_outputs, first_references, second_references, third_references = [], [], [], []
    grouped_df = df.groupby(by=['meta'])
    for group_name, group in grouped_df:
        selected_choices = [0, 1, 2]
        system_outputs.append(group.iloc[0]['prep_predicted_answer'])
        first_references.append(group.iloc[selected_choices[0]]['prep_gold_answer'])
        second_references.append(group.iloc[selected_choices[1]]['prep_gold_answer'])
        third_references.append(group.iloc[selected_choices[2]]['prep_gold_answer'])

    os.system(f'mkdir -p '+temp_dir)

    first_references_file = open(temp_dir+'/references-0', 'w+')
    second_references_file = open(temp_dir+'/references-1', 'w+')
    third_references_file = open(temp_dir+'/references-2', 'w+')
    candidates_file = open(temp_dir+'/candidates', 'w+')

    for (first_reference, second_reference, third_reference, candidate) in zip(first_references, second_references, third_references, system_outputs):
        first_references_file.write(first_reference+'\n')
        second_references_file.write(second_reference+'\n')
        third_references_file.write(third_reference+'\n')
        candidates_file.write(candidate+'\n')

    first_references_file.close()
    second_references_file.close()
    third_references_file.close()
    candidates_file.close()

    references = [first_references, second_references, third_references]

    selected_choices = [0, 1, 2]

    all_bleurt_scores = []
    for choice in selected_choices:
        reference_set = references[choice]
        bleurt_command = f"python -m bleurt.score -candidate_file={temp_dir}/candidates -reference_file={temp_dir}/references-{choice} -bleurt_checkpoint=bleurt/bleurt-base-128 -scores_file={temp_dir}/scores-{choice}"

        os.system(bleurt_command)

        bleurt_scores = []
        with open(temp_dir+"/scores-"+str(choice)) as fp:
            for line in fp:
                score = float(line.strip())
                bleurt_scores.append(score)
        all_bleurt_scores.append(bleurt_scores)

    max_bleurt_scores = []
    for i in range(len(all_bleurt_scores[0])):
        max_val = all_bleurt_scores[0][i]
        for j in range(3):
            if all_bleurt_scores[j][i] > max_val:
                max_val = all_bleurt_scores[j][i]
        max_bleurt_scores.append(max_val)

    bleurt_score = sum(max_bleurt_scores) / len(max_bleurt_scores)

    del_bleurt_command = 'rm -rf '+temp_dir
    os.system(del_bleurt_command)

    return bleurt_score, np.array(max_bleurt_scores)

# from comments in https://www.saltycrane.com/blog/2008/01/how-to-find-intersection-and-union-of/
# the following four functions allow calculation of union and intersection of two lists even if elements occur multiple times in a list
def to_multiset(x):
    result = set()
    max_rep = len(x)
    for elt in x:
        for n in range(max_rep):
            n_elt = (elt,n)
            if n_elt not in result:
                result.add(n_elt)
                break
    return result

def from_multiset(x):
    return sorted([elt for elt,n in x])

def multi_union(a, b):
    aa = to_multiset(a)
    bb = to_multiset(b)
    return from_multiset(aa | bb)

def multi_intersect(a, b):
    aa = to_multiset(a)
    bb = to_multiset(b)
    return from_multiset(aa & bb)

def unigram_overlap(answer, influencer, setop=False):

    """
    This function calculates the amount of overlap between the answer and either the narrative or the original sentence from which the question was derived from.
    It is normalized by the length of the answer
    influencer can be either narrative or sentence from which question was generated
    if setop is true, it will return result of set operation calculation
    """

    if type(answer) != str:
        return 0.0
    answer_tokens = answer.split()
    if len(answer_tokens) == 0:
        return 0.0
    answer_tokens_set = set(answer_tokens)
    influencer_tokens = influencer.split()
    influencer_tokens_set = set(influencer_tokens)
    if setop:
        token_set_intersection_count = len(list(answer_tokens_set.intersection(answer_tokens_set)))
        token_set_union_count = len(list(answer_tokens_set.union(answer_tokens_set)))
        return token_set_intersection_count / len(list(answer_tokens_set))
    else:
        token_union_count = len(multi_union(answer_tokens, influencer_tokens))
        token_intersection_count = len(multi_intersect(answer_tokens, influencer_tokens))
        return token_intersection_count / len(answer_tokens)

def initialise_df_for_single_reference_bleu(predictions, references):
    df = pd.DataFrame()
    df['gold_answer'] = references
    df['predicted_answer'] = predictions
    return df

def judge_majority_question_answerable(df):

    """
    Judge whether a question has been marked answerable by majority
    """

    grouped_df = df.groupby(by=['meta'])
    answerability_map = {}
    for group_name, group in grouped_df:
        answerable_score = 0
        for idx, row in group.iterrows():
            if row['is_ques_answerable'] == 'Not Answerable':
                answerable_score -= 1
            else:
                answerable_score += 1
        if answerable_score > 0:
            answerability_map[group_name] = 'Answerable'
        else:
            answerability_map[group_name] = 'Not Answerable'
    df['majority_answerable_judgment'] = df['meta'].map(answerability_map)
    return df

def perform_evaluation(df, temp_dir, args):
    
    """
    Run all metrics on the dataframe. In order, they are ROUGE, BLEU, BertScore and BLEURT.
    It also returns sentence pair wise scores for each metric.
    """

    source_multi_ref_dict = create_multi_reference_dictionary_for_gold_sentences(df)
    references, hypotheses = create_inputs_for_rouge(df, source_multi_ref_dict)
    rouge(hypotheses, references)

    grouped_df = df.groupby(by=['meta'])
    predictions, first_references, second_references, third_references = [], [], [], []
    meta_list = []
    for group_name, group in grouped_df:
        selected_choices = [0, 1, 2]
        meta_list.append(group_name)
        predictions.append(group.iloc[0]['predicted_answer'])
        first_references.append(group.iloc[selected_choices[0]]['gold_answer'])
        second_references.append(group.iloc[selected_choices[1]]['gold_answer'])
        third_references.append(group.iloc[selected_choices[2]]['gold_answer'])
    reference_lists = [first_references, second_references, third_references]
    
    all_refs = []
    for ref in reference_lists:
        all_refs.extend(ref)
    
    #  idf should be calculated over all gold answers
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, idf=True, idf_sents=all_refs)

    # one_ref_bleus will store 3 arrays, one related to each reference set
    # each value in an any array corresponds to one gold answer and one model answer
    print('BLEU calculation')
    one_ref_bleus = []
    for idx, reference_list in enumerate(reference_lists):
        bleu_df = initialise_df_for_single_reference_bleu(predictions, reference_list)

        sentence_level_bleus = sentence_level_single_reference_bleu_scores(predictions, reference_list)
        one_ref_bleus.append(sentence_level_bleus)
    
    max_bleu_scores = []
    for i in range(len(one_ref_bleus[0])):
        max_val = one_ref_bleus[0][i]
        for j in range(3):
            if one_ref_bleus[j][i] > max_val:
                max_val = one_ref_bleus[j][i]
        max_bleu_scores.append(max_val)
    new_calc_max_bleu = sum(max_bleu_scores) / len(max_bleu_scores)

    log_str = f'Max BLEU score is: {round(new_calc_max_bleu,2)}'
    print(log_str)
    logging.info(log_str)

    print('BertScore calculation')

    max_multibertscore_scores, metas, real_score = sentence_level_multi_bertscore(df, scorer)
    new_calc_max_bertscore = sum(max_multibertscore_scores) / len(max_multibertscore_scores)

    log_str = f'Max BertScore score is: {round(new_calc_max_bertscore,4)}'
    print(log_str)
    logging.info(log_str)

    print('BLEURT calculation')
    max_bleurt_score, bleurt_scores = single_reference_bleurt(df, temp_dir+'-bleurt')
    log_str = f'Max BLEURT score is: {round(max_bleurt_score,4)}'
    print(log_str)
    logging.info(log_str)

    return max_multibertscore_scores, max_bleu_scores, meta_list, bleurt_scores

def main(args):
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG, format='')
    start_time = time.ctime()
    logging.info(f'Starting at {start_time}')
    logging.info(vars(args))
    log_str = f'Input file to evaluate: {args.test_output_file}'
    logging.info(log_str)
    if not args.temp_dir:
        print("Please provide temp dir name/path for BLEURT calculation")
        sys.exit()
    df = pd.read_csv(args.test_output_file)

    # the following replaces any NaN style answers in the file (model generates empty string, pandas writes it as NaN) with an empty string
    # this is possible for GPT-2 style systems and does not affect scores
    df['predicted_answer'] = df['predicted_answer'].fillna('')

    df['narrative_lexical_overlap'] = df.apply(lambda row: unigram_overlap(row['predicted_answer'], row['narrative']), axis=1)
    df['mean_narrative_lexical_overlap'] = df['meta'].map(df.groupby('meta')['narrative_lexical_overlap'].mean())
    predictions_mean_narrative_lexical_overlap = 100*df['mean_narrative_lexical_overlap'].sum()/df.shape[0]
    log_str = f'Average narrative lexical overlap for predicted answers is {predictions_mean_narrative_lexical_overlap}'
    logging.info(log_str)

    if args.temp_dir[-1] == '/':
        temp_dir = args.temp_dir[:-1]
    else:
        temp_dir = args.temp_dir

    max_bert_scores, max_bleu_scores, meta_list, max_bleurt_scores = perform_evaluation(df, temp_dir+'-full', args)

    try:
        implicit_df = df[df['is_ques_answerable'] == 'Not Answerable']
        log_str = f'Now evaluating on implicit answer questions'
        logging.info(log_str)
        _, _, _, _ = perform_evaluation(implicit_df, temp_dir+'-implicit', args)
    except:
        pass

    end_time = time.ctime()
    logging.info(f'Ended at {end_time}')

if __name__ == '__main__':
    args = parse_args()
    main(args)