"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

template = {
    #"prompt_no_input": "{instruction}### Output:\n",
    "prompt_no_input": "{instruction}",
    "response_split": "### Output:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        
        res = self.template["prompt_no_input"].format(
            instruction=instruction
        )
        
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
