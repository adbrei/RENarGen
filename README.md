# RENarGen
Codebase for RENarGen  (NAACL '24)

**Paper:**

**Abstract:**  Human writers often *bookend* their writing with endings related to the beginnings in order to compose a satisfying narrative that ''closes the loop.'' Motivated by this observation, we propose RENarGen, a controllable story-generation paradigm that generates narratives by first ensuring the first and last sentences are related and then infilling middle sentences. Our contributions include an initial exploration of how bookending affects language modeling for stories. Automatic and human evaluations indicate RENarGen produces stories with more narrative closure than current autoregressive models.

## Datasets

We use [ROCStories corpus](https://cs.rochester.edu/nlp/rocstories/). For training, we combine Spring 2016 and Winter 2017 sets and split 80:20 for training and validation. For evaluation we use the first sentences from Cloze Spring 2016. 

For training a sentence relatedness model, we use [STR-2022](https://github.com/Priya22/semantic-textual-relatedness/blob/master/sem_text_rel_ranked.csv).

## RENarGen for LMs

To set up RENarGen-LM:
1. Obtain the ROCStories and STR-2022 corpora. Put CSV files in ``data`` directory.
2. Train models by running each cell block in ``setup.ipynb``
3. Generate stories by running ``run.py``

# RENarGen for LLMs

Our methods are model agnostic. For our experiments, we use Llama2-7b-chat and Llama2-70b-chat. To run Llama2 locally, we recommend using the [Ollama framework](https://github.com/ollama/ollama). However, users may alternatively use cloud-based solutions or any other models to which they have access.

All prompts (including system and model prompts) are located in `RENarGen/RENarGen-LLM` directory.
