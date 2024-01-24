# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.
TODO: Write a Short Description of the task.
Homepage: TODO: Add the URL to the task's Homepage here.
"""
from abc import ABC

from lm_eval.base import Task
from evaluate import load
# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class Leetcode_en_one_shot(Task):
     # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    # DATASET_PATH = "zhangzhang9999/code"
    # DATASET_PATH = "zhangzhang9999/dataset"
     #TODO:私人数据集
    #DATASET_PATH = "zhangzhang9999/data_ehinese"
    DATASET_PATH = "zhangzhang9999/one_shot_en_difficult"
    #DATASET_PATH = "zhangzhang9999/difficult_ch"
    #TODO!!:中文变成json类型的编码
    #DATASET_PATH = "zhangzhang9999/difficult_ch1"
    #DATASET_PATH = "zhangzhang9999/difficult_ch1"

    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
                stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"],
                requires_execution=True,
            )
    def get_dataset(self):
        # print(self.dataset['test'])
        print(self.dataset)
        #TODO!!:print(self.dataset["train"])改了
        print(self.dataset["train"])
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        print(doc["prompt"])
        return doc["prompt"].strip()

    def get_reference(self, doc):
        # print(doc)
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        #TODO!!:这里改了代码，dataset["test"] 改为dataset["train"],11.10改为dataset["test"]
        prompt = self.get_prompt(self.dataset["train"][idx])
        generation = generation[len(prompt):]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
