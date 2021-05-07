"""
This script is used to generate the input for a model's human evaluation - judging the validity of model answers given a question and story
"""
import os
import sys
import csv
import json
import argparse
import pandas as pd
import random

random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='CSV file containing model output')
    parser.add_argument('--task-input-file-prefix', type=str, help='Prefix for human evaluation task input CSV file to be generated')
    args, _ = parser.parse_known_args()
    return args

def initialise_output_file_writer(task_input_file_prefix, count):

    """
    count is the current number of output file being generated i.e. first, second etc
    """

    out_fp = open(task_input_file_prefix+'_'+str(count)+'.csv', 'w+')
    fieldnames = ["narrative_meta", "narrative", "question_meta", "question", 
                 "ques_answer", "is_ques_answerable"]
    writer = csv.DictWriter(out_fp, fieldnames, lineterminator='\n', quoting=csv.QUOTE_ALL)
    writer.writeheader()
    return out_fp, writer

def create_mturk_input_csv_for_model_answers(df, task_input_file_prefix):

    """
    Create an Mturk input file with the fields: narrative_meta,narrative,question_meta,question,ques_answer

    MTurk only allows 500 lines of input for one HIT. This function creates csv files with 500 records each.
    If there are more than 500 lines of input overall, it will create multiple files.
    """

    grouped_df = df.groupby(by='question_meta')
    num_rows = 1
    file_count = 1
    out_fp, writer = None, None
    for group_name, group in grouped_df:
        info = {}
        if num_rows % 500 == 1:
            if num_rows != 1:
                out_fp.close()
            out_fp, writer = initialise_output_file_writer(task_input_file_prefix, file_count)
            file_count += 1
        for row_idx, row in group.iterrows():
            info['narrative_meta'] = row['narrative_meta']
            info['narrative'] = row['narrative']
            info['question_meta'] = group_name
            info['question'] = row['question_text']
            info['ques_answer'] = row['answer']
            info['is_ques_answerable'] = row['is_ques_answerable']
        num_rows += 1
        writer.writerow(info)
    out_fp.close()

def create_human_eval_input(args):

    """
    This function will create human evaluation task input using model outputs.
    """

    input_df = pd.read_csv(args.input_file)
    input_df = input_df.drop_duplicates(subset=['question_meta'])
    input_df = input_df[['narrative', 'question', 'question_meta', 'predicted_answer', 'is_ques_answerable']]
    input_df =  input_df.rename(columns={'question': 'question_text', 'predicted_answer': 'answer'})
    narrative_metas  = {}
    for idx, row in input_df.iterrows():
        narrative_metas[row['question_meta']] = '_'.join(row['question_meta'].split('_')[:3])
    input_df['narrative_meta'] = input_df['question_meta'].map(narrative_metas)
    create_mturk_input_csv_for_model_answers(input_df, args.task_input_file_prefix)

def main(args):
    create_human_eval_input(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)