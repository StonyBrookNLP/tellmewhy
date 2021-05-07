"""
This script is used to process the output for human evaluation task - judging the validity of answers given a question and story
"""
import os
import sys
import csv
import json
import argparse
import math
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='CSV output of human evaluation task')
    args, _ = parser.parse_known_args()
    return args

def fleiss_kappa_tests():

    """
    Tests from Wikipedia page - https://en.wikipedia.org/wiki/Fleiss%27_kappa - to make sure implementation is correct
    """

    annotations_matrix = [[0, 0, 0, 0, 14],
                          [0, 2, 6, 4, 2],
                          [0, 0, 3, 5, 6],
                          [0, 3, 9, 2, 0],
                          [2, 2, 8, 1, 1],
                          [7, 7, 0, 0, 0],
                          [3, 2, 6, 3, 0],
                          [2, 5, 3, 2, 2],
                          [6, 5, 2, 1, 0],
                          [0, 2, 2, 3, 7]]

    gold_fleiss_kappa = 0.21

    calculated_fleiss_kappa = round(fleiss_kappa(annotations_matrix), 2)
    assert calculated_fleiss_kappa == gold_fleiss_kappa

def weighted_fleiss_kappa(new_final_labels, weights, weighted=True):
    
    """
    Code from Mohaddeseh's COLING paper
    """

    table = 1 * np.asarray(new_final_labels)   # avoid integer division

    n_sub, n_cat =  table.shape

    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    # assume fully ranked
    assert n_total == n_sub * n_rat

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    if weighted:
        table_weight = 1 * np.asarray(weights)     
        table2 = np.matmul(table , table_weight)
        table2 = np.multiply(table2,table)
    else:
        table2 = table * table
   
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    p_mean_exp = (p_cat*p_cat).sum()
  
    kappa = float(p_mean - p_mean_exp) / (1- p_mean_exp)

    return round(kappa, 4)

def fleiss_kappa(M):

    """
    From: https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3
    Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    # turned out to be same as my implementation
    """

    M = np.array(M)
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)

def extract_true_annotations(df, feature, num_categories=5):
    
    """
    This function extracts the annotators' judgments for a particular feature.
    The new dataframe helps calculate Fleiss Kappa
    """

    if feature == 'validity':
        df = df[['answer_meta', 'is_ans_valid']]
        df = df.rename(columns={'is_ans_valid': 'feature'})
    elif feature == 'grammaticality':
        df = df [['answer_meta', 'is_ans_grammatical']]
        df = df.rename(columns={'is_ans_grammatical': 'feature'})
    grouped_df = df.groupby(by=['answer_meta'])
    annotations_matrix = []
    for group_name, group in grouped_df:
        annotations = [0] * num_categories
        for idx, row in group.iterrows():
            if pd.isna(row['feature']):
                annotations[0] += 1
            else:
                annotations[int(row['feature'])+2] += 1
        annotations_matrix.append(annotations)

    return annotations_matrix

def extract_binary_annotations(df, feature):
    
    """
    This function extracts the annotators' judgments for a particular feature.
    The new dataframe helps calculate Fleiss Kappa
    The true annotations will be transformed into binary form
    """

    if feature == 'validity':
        df = df[['answer_meta', 'is_ans_valid']]
        df = df.rename(columns={'is_ans_valid': 'feature'})
    elif feature == 'grammaticality':
        df = df [['answer_meta', 'is_ans_grammatical']]
        df = df.rename(columns={'is_ans_grammatical': 'feature'})
    grouped_df = df.groupby(by=['answer_meta'])
    annotations_matrix = []
    for group_name, group in grouped_df:
        annotations = [0] * 3
        for idx, row in group.iterrows():
            if pd.isna(row['feature']):
                annotations[0] += 1
            else:
                if int(row['feature']) > 0:
                    annotations[2] += 1
                elif int(row['feature']) < 0:
                    annotations[0] += 1
                else:
                    annotations[1] += 1
        annotations_matrix.append(annotations)

    return annotations_matrix

def majority_judgment(df, feature):
    
    if feature == 'validity':
        df = df[['answer_meta', 'is_ans_valid']]
        df = df.rename(columns={'is_ans_valid': 'feature'})
    elif feature == 'grammaticality':
        df = df [['answer_meta', 'is_ans_grammatical']]
        df = df.rename(columns={'is_ans_grammatical': 'feature'})
    grouped_df = df.groupby(by=['answer_meta'])
    annotation_scores = []
    for group_name, group in grouped_df:
        annotation_score = 0
        for idx, row in group.iterrows():
            if pd.isna(row['feature']):
                annotation_score -= 1
            else:
                if int(row['feature']) > 0:
                    annotation_score += 1
                elif int(row['feature']) < 0:
                    annotation_score -= 1
        annotation_scores.append(annotation_score)
    percentage = len([i for i in annotation_scores if i > 0]) / len(annotation_scores)
    return round(percentage * 100, 2), len(annotation_scores)

def find_annotator_disagreements(df, feature):

    """
    This function extracts cases where annotators severely disagree in their judgments.
    Polarising examples are extracted to be analysed in this function
    e.g. 2 annotators score it >=+1 and one scores it <=-1
    extreme disagreement is cases where one annotator marks it -2 and the others >=+1 or vice versa
    """

    clean_df = df.drop_duplicates(subset=['answer_meta'])
    annotations_col_name = feature+'_annotations'
    disagreement_judgments, extreme_disagreement_judgments = {}, {}
    for row_idx, row in df.iterrows():
        key = row['answer_meta']
        pos_count = len([i for i in row[annotations_col_name] if i > 0])
        neg_count = len([i for i in row[annotations_col_name] if i < 0])

        # normal disagreements
        if (pos_count > neg_count) and (neg_count > 0):
            disagreement_judgments[key] = True
        elif (neg_count > pos_count) and (pos_count > 0):
            disagreement_judgments[key] = True
        else:
            disagreement_judgments[key] = False

        # extreme disagreement
        if -2 in row[annotations_col_name]:
            annotations_copy = row[annotations_col_name]
            # remove -2 from that list
            del annotations_copy[row[annotations_col_name].index(-2)]
            # check if remaining two elements are both positive
            if len([i for i in annotations_copy if i > 0]) == 2:
                extreme_disagreement_judgments[key] = True
            else:
                extreme_disagreement_judgments[key] = False
        elif 2 in row[annotations_col_name]:
            annotations_copy = row[annotations_col_name]
            # remove 2 from that list
            del annotations_copy[row[annotations_col_name].index(2)]
            # check if remaining two elements are both negative
            if len([i for i in annotations_copy if i < 0]) == 2:
                extreme_disagreement_judgments[key] = True
            else:
                extreme_disagreement_judgments[key] = False
        else:
            extreme_disagreement_judgments[key] = False

    num_disagreements, num_extreme_disagreements = 0, 0
    for k, v in disagreement_judgments.items():
        if v == True:
            num_disagreements += 1
    for k, v in extreme_disagreement_judgments.items():
        if v == True:
            num_extreme_disagreements += 1

    df[feature+'_disagreement'] = df['answer_meta'].map(disagreement_judgments)
    df[feature+'_extreme_disagreement'] = df['answer_meta'].map(extreme_disagreement_judgments)

    print(f'Number of annotator disagreements is {num_disagreements} out of {clean_df.shape[0]}')
    print(f'Number of extreme annotator disagreements is {num_extreme_disagreements} out of {clean_df.shape[0]}')

def add_annotations(df, feature):

    """
    This function add all the annotations for a particular question to each record associated with it
    """

    if feature == 'validity':
        col_name = 'is_ans_valid'
    elif feature == 'grammaticality':
        col_name = 'is_ans_grammatical'
    annotations = {}
    grouped_df = df.groupby(by='answer_meta')
    for group_name, group in grouped_df:
        answer_annotations = []
        for row_idx, row in group.iterrows():
            answer_annotations.append(row[col_name])
        annotations[group_name] = answer_annotations
    df[feature+'_annotations'] = df['answer_meta'].map(annotations)

    return df

def analyse_model_output_validity(df):

    """
    This function reads the annotated output file for model output through MTurk.
    """

    df = df.drop(columns=['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Approve', 'Reject', 'Answer.Comments'])
    df = df.rename(columns={'Input.narrative_meta': 'narrative_meta', 'Input.question_meta': 'question_meta', 'Input.question': 'question_text', 'Input.narrative': 'narrative_text', 'Input.ques_answer': 'answer', 'Answer.is_ques_ans0_grammatical': 'is_ans_grammatical', 'Answer.is_ques_ans0_valid': 'is_ans_valid', 'Input.is_ques_answerable': 'is_ques_answerable'})
    df['answer_meta'] = df['question_meta'] + '_answer'

    return df

def evaluation(df):

    """
    This function performs all the evaluation required
    """

    majority_grammaticality_decision, num_questions = majority_judgment(df, 'grammaticality')
    print(f'Annotators agree (by majority) that {majority_grammaticality_decision}% out of {num_questions} answers are grammatical for their respective questions')
    
    majority_validity_decision, num_questions = majority_judgment(df, 'validity')
    print(f'Annotators agree (by majority) that {majority_validity_decision}% out of {num_questions} answers are valid for their respective questions')

    grammaticality_annotations_matrix = extract_true_annotations(df, 'grammaticality', 5)

    validity_annotations_matrix = extract_true_annotations(df, 'validity', 5)

    grammaticality_binary_annotations_matrix = extract_binary_annotations(df, 'grammaticality')

    validity_binary_annotations_matrix = extract_binary_annotations(df, 'validity')

    print(f'Grammaticality:')
    find_annotator_disagreements(df, 'grammaticality')

    print(f'Validity:')
    find_annotator_disagreements(df, 'validity')

    # weights from Mohaddeseh's COLING paper
    true_annotation_weights = np.array([[1, math.cos(math.pi/8), math.cos(math.pi/4), math.cos(3*math.pi/8), 0],
                                        [math.cos(math.pi/8), 1, math.cos(math.pi/8), math.cos(math.pi/4), math.cos(3*math.pi/8)],
                                        [math.cos(math.pi/4), math.cos(math.pi/8), 1, math.cos(math.pi/8), math.cos(math.pi/4)],
                                        [math.cos(3*math.pi/8), math.cos(math.pi/4), math.cos(math.pi/8), 1, math.cos(math.pi/8)],
                                        [0, math.cos(3*math.pi/8), math.cos(math.pi/4), math.cos(math.pi/8), 1]])
    weighted_grammaticality_kappa = weighted_fleiss_kappa(grammaticality_annotations_matrix, true_annotation_weights)
    print(f'Grammaticality true weighted kappa: {weighted_grammaticality_kappa}')

    weighted_validity_kappa = weighted_fleiss_kappa(validity_annotations_matrix, true_annotation_weights)
    print(f'Validity true weighted kappa: {weighted_validity_kappa}')

    # weights created using same intuition as above
    binary_annotation_weights = np.array([[1, math.cos(math.pi/8), math.cos(math.pi/4)],
                                          [math.cos(math.pi/8), 1, math.cos(math.pi/8)],
                                          [math.cos(math.pi/4), math.cos(math.pi/8), 1]])
    weighted_grammaticality_binary_kappa = weighted_fleiss_kappa(grammaticality_binary_annotations_matrix, binary_annotation_weights)
    print(f'Grammaticality binary weighted kappa: {weighted_grammaticality_binary_kappa}')

    weighted_validity_binary_kappa = weighted_fleiss_kappa(validity_binary_annotations_matrix, binary_annotation_weights)
    print(f'Validity binary weighted kappa: {weighted_validity_binary_kappa}')

def main(args):
    df = pd.read_csv(args.input_file)

    df = analyse_model_output_validity(df)

    num_unique_records = df['answer_meta'].nunique()
    print(f'Total number of questions: {num_unique_records}')
    num_unique_ques = df['question_meta'].nunique()
    print(f'Number of unique questions: {num_unique_ques}')
    
    df = add_annotations(df, 'grammaticality')
    df = add_annotations(df, 'validity')

    fleiss_kappa_tests()

    evaluation(df)

    print('Now evaluating on implicit-answer questions')

    df = df[df['is_ques_answerable'] == 'Not Answerable']

    evaluation(df)

if __name__ == '__main__':
    args = parse_args()
    main(args)