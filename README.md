# DocuMint: Evaluation of Automated Docstring Generation using Small Language Models

## Background
Large Language Models (LLMs) are having their moment right now. The latest trends in LLMs however, are Small Language Models (generally 13B parameters or less) such as Mistral, Gemma, and Llama family of models. They offer lower inference time, cost and can be deployed locally in consumer devices. 

## Research
Our study is to explore if we can leverage them to automatically generate docstring (on classes, functions, etc.) given a Python file. This docstring can then be used to generate documentation of a codebase. This automates an important step in the software development pipeline i.e., the documentation step.

## Data
Extract well documented python files (proper docstring) from World of Code. Split the data into fine-tuning and validation.

World of Code is a large dataset, so we need to mine for the information that we're looking for:
1. Look at commits that have “added docstring” in the commit message.
2. Look at blobs, not just files.
3. Look for projects that list documentation links in the README (parse the README).

## Methodology
1. Perform human evaluation (preference ranking) on the generated docstring.
2. Evaluate the “emergent behaviors" in generating docstring.
   - Does improving the size of the model improve the quality of docstring?
   - Conduct a user preference study (human evaluation)
4. Fine tuning vs. base model.
   - Does the fine tuning step help?
   - NOTE: This will be based on time contraints, may not be feasible short-term.

## People
Shelah Ameli (@ShelahAmeli)

Adam Cook (@ajcook247 / acook46@vols.utk.edu)

Bibek Poudel (@poudel-bibek)

Sekou Traore (@Sekou2077)
