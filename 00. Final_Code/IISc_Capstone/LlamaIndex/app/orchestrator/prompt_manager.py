# app/orchestrator/prompt_manager.py
import os

class PromptManager:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir

    def load(self, name: str) -> str:
        path = os.path.join(self.template_dir, f"{name}.txt")
        return open(path).read()

    def format(self, template_name: str, **kwargs) -> str:
        tmpl = self.load(template_name)
        # Ensure every placeholder is at least the empty string
        for var in ["user_message","context","intent","sentiment","api_result"]:
            if var not in kwargs or kwargs[var] is None:
                kwargs[var] = ""
        # Now do replacements
        for k, v in kwargs.items():
            tmpl = tmpl.replace(f"{{{k}}}", v)
        return tmpl
