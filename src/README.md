# Running Human Evaluation

Due to the open-ended nature of the task, we posit that human evaluation is the right way to evaluate performance on the why question answering task.
We provide the evaluation template as well as related scripts so that the community can analyse model performance easily.

## Files

 - `human_evaluation.html` - This is the MTurk template for the human evaluation of model answers.
 - `create_human_evaluation_input.py` - This script takes a model prediction file and creates the requisite input CSV for the MTurk human evaluation.
 - `analyse_human_evaluation_output.py` - This script takes the raw output of the MTurk human evaluation and calculates grammaticality and validity statistics for model answers (and inter-annotator agreement for both) based on the obtained annotations.
 - `automatic_evaluation.py` - This script takes the predictions of any model over the full test set in CSV format and performs evaluation (BLEU, BLEURT, BertScore) as described in the [TellMeWhy](https://www3.cs.stonybrook.edu/~ylal/files/papers/acl2021F.pdf) paper.
 - `dummy_human_evaluation_input.csv` - This acts as a dummy predictions file. Your automatic evaluation input file should mimic its structure.
 - `dummy_human_evaluation_input.csv` - This acts as a dummy MTurk input file. Your human evaluation HIT input file should mimic its structure.

## Human Evaluation Steps

Please follow the given steps to deploy human evaluation and analyse the results.

1. Install the requirements using: `pip install -r requirements.txt` using `Python 3.7.11`.
2. Perform inference on the `test_annotated` split and save the results to a CSV file. 
For best results, use the `csv.writer` object to write your predictions to a file.
Your predictions file should contain the following columns: 'narrative', 'question', 'question_meta', 'predicted_answer', 'is_ques_answerable'.
Please add your model's inference in the `predicted_answer` column and retain every other listed column from the dataset file.
3. Create a new HIT using your Amazon Mechanical Turk account.
Please use the following properties:
   - Reward per response: $0.03 (please re-calculate according to minimum wage requirements in your area)
   - Number of respondents: 3
   - Time allotted per Worker: 1 hour
   - Survey expires in: 7 days
   - Require that Workers be Masters to do your tasks: Yes
   - Task visibility: Public
4. Create MTurk HIT inputs using `create_human_evaluation_input.py`. <br>
Example usage: `python create_human_evaluation_input.py --input-file predictions.csv --task-input-file-prefix task_input`. <br>
This will create a file named `task_input_1.csv`.
You can use this as input for human evaluation of your model's predictions.
5. After the evaluation is complete, you can use the raw output from MTurk with `analyse_human_evaluation_output.py` to obtain performance statistics.
This also calculates weighted Fleiss Kappa to gauge inter-annotator agreement. <br>
Example usage: `python analyse_human_evaluation_output.py --input-file mturk_output.csv`.

This results in numbers similar to the ones used for Figure 2 in the [TellMeWhy](https://www3.cs.stonybrook.edu/~ylal/files/papers/acl2021F.pdf) paper.

## Automatic Evaluation Steps

Please follow the given steps to deploy human evaluation and analyse the results.

1. Install the requirements using: `pip install -r requirements.txt` using `Python 3.7.11`.
2. Perform inference on the `test_full` split and save the results to a CSV file.
For best results, use the `csv.writer` object to write your predictions to a file.
Your predictions file should contain the following columns: 'narrative', 'question', 'meta', 'predicted_answer', 'gold_answer', is_ques_answerable'.
Please add your model's inference in the `predicted_answer` column.
3. Run the `automatic_evaluation.py` script with the required arguments.
Example usage: `python automatic_evaluation.py --test-output-file test_full_model_predictions.csv --log-file test_full_model.auto_metrics --temp-dir temp-dir-name`.

This results in numbers similar to the ones used for Table 4 in the [TellMeWhy](https://www3.cs.stonybrook.edu/~ylal/files/papers/acl2021F.pdf) paper.

## Contact

Please reach out to [Yash Kumar Lal](mailto:ylal@cs.stonybrook.edu) if you face any issues.
If you want your model's result to be displayed on the main webpage, please send us the raw MTurk output file you obtain.