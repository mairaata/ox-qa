import pandas as pd
from google.colab import drive
from typing import List, Dict
import os

class EvaluationComponent:

    def __init__(self,documents_path, question_path,output_path,client_details):
        self.client_details = client_details
        self.rag  = LlamaChatRAG(documents_path)
        self.question_df = pd.read_excel(question_path)
        self.documents_path = documents_path
        self.output_path = output_path


    def run_evaluation(self):

        predictions = []
        MANAGER_OR_ADVISOR = self.client_details['MANAGER_OR_ADVISOR']
        FUND = self.client_details['FUND']

        for idx, row in self.question_df.iterrows():
            question = str(row['questions'])
            if type(question) is str:
              question = question.replace("MANAGER_OR_ADVISOR", MANAGER_OR_ADVISOR)
            else:
              print(question)

            actual = row['actual_answers']

            # Run RAG pipeline
            result = self.rag.answer(question)
            predicted=result["result"]

            # Compile for output
            entry = {
                'id': idx,
                'question': question,
                'predicted_answers': predicted,
                'actual_answers': actual,
                'result':result
            }
            predictions.append(entry)

        # Save output
        out_df = pd.DataFrame(predictions)
        self.save(out_df)

    def save(self, out_df):
        filename = self.client_details['client_name'] + '_llama2_evaluation_output.csv'
        out_df.to_csv(os.path.join(self.output_path,filename))