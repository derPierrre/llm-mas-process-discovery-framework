import os
from typing import Dict, List, Any, Optional
import re

def get_prompt(name: str, replacements: Dict[str, str] = dict(), already_processed: Optional[List[str]] = None) -> str:
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the prompt file
    prompt_path = os.path.join(script_dir, f"prompts/{name}.txt")
    
    with open(prompt_path, "r") as file:
        prompt_text = file.read()

    # Check for any {{file_name}} references in the prompt
    file_refs = re.findall(r'\{\{([^}]+)\}\}', prompt_text)
    
    # Load each referenced file and replace the {{file_name}} with its content
    for file_ref in file_refs:
        if file_ref == name:
            raise ValueError(f"Prompt '{name}' cannot reference itself.")

        if already_processed is None:
            already_processed = []

        if file_ref in already_processed:
            raise ValueError(f"Prompt '{name}' contains a circular reference to '{file_ref}'")

        already_processed.append(file_ref)

        file_content = get_prompt(file_ref, replacements, already_processed.copy())
        
        replace = '{{' + file_ref + '}}'
        prompt_text = prompt_text.replace(replace, file_content)

    return prompt_text.format(**replacements)