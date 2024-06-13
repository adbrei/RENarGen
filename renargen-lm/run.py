from RENarGen import RENarGen
import pandas as pd
from tqdm import tqdm
from random import randint

print('Starting...')

# Import data
DATA_PATH = 'cloze_test_spring2016.csv'
data = pd.read_csv(DATA_PATH, usecols=['InputSentence1'])
starts = data.InputSentence1.to_list()

print('Number samples being generated: {}'.format(len(starts)))

#####
# TODO: Choose stop model from {phrase2stop_gpt, stop_baseline}
# TODO: Choose story infiller from {infill_generator, a2_1, a2_3}
stop_model = 'phrase2stop_gpt'         # default
infilling_model = 'infill_generator'   # default

# RENarGen model
gen = RENarGen(
   stop_model_type=stop_model, 
   infilling_model_type=infilling_model,
   interactive=False
   ) 

# Inference
renargen, renargen_ordered, phrases_list = [], [], []

for s in tqdm(starts):
   
   # Samples from RENarGen
   gen.reset()
   # num_sentences = randint(5, 10)
   num_sentences = 5             # Total sentences
   story, num_sentences, story_ordered, phrases = gen.build_story(s, num_s=num_sentences-2)
   renargen.append(story)
   # renargen_ordered.append(story_ordered + f' ({num_sentences})')
   # phrases_list.append(phrases)

   print(phrases)
   print(story)

results = pd.DataFrame()
results['start'] = starts
results['renargen'] = renargen
# results['renargen_ordered'] = renargen_ordered
# results['phrases'] = phrases_list

results.to_csv('results.csv', index=False)
