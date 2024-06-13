# Standard libraries
import re
from nltk.tokenize import sent_tokenize
import random

# Language models & pipelines
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline

# Attach to GPU
# import torch
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'Connected to GPU --> {device}')


class RENarGen():

    def __init__(self, stop_model_type='phrase2stop_gpt', infilling_model_type='infill_generator', interactive=False):

        self.interactive = interactive                      # Boolean for controllability features
        self.stop_model_type = stop_model_type              # Stop model
        self.infilling_model_type = infilling_model_type    # Infilling model
        self.sentences = []                                 # store sentences as list to easily infill
        self.sentences_with_order = []                      # list with labeled sentences (infill order)
        self.phrases_str = ''                               # string list of phrases
        
        print('\n', '-'*10)
        print(f'Controllability mode on: {self.interactive}')
        print(f'Stop Model Type: {self.stop_model_type}')
        print(f'Infilling Model Type: {self.infilling_model_type}\n')
    
        ### Load models
        print("Loading models...\n")

 
        PATHS = {
            # RENarGen has two main components: the stop generator and the story infiller
            # 1. Stop generators
            'stop_baseline'     : 'models/stop_baseline',       # GPT-2 baseline model          (given start, generates stop)
            'phrase_generator'  : 'models/phrase_generator',    # GPT-2 phrase list generator   (given start, generates list of phrases)
            'stop_generator'    : 'models/stop_generator',      # GPT-2 stop generator          (given start + rep phrases, generates stop)

            # 2. Story infillers
            'position_classifier'   : 'models/position_classifier', # BERT classifier to determine where to infill next
            'infill_generator'      : 'models/infill_generator',    # GPT-2 model generates an infill sentence

            'a2_1'                  : 'models/ablation2_1',         # Remove position classifer; infill 3 sentences
            'a2_3'                  : 'models/ablation2_3'          # Remove phrase generator
        }
    
        ### Stop generator

        # Load GPT-2 phrase generator / stop generator system
        if stop_model_type == 'phrase2stop_gpt':

            # Load phrase list generator model
            self.phrases_tokenizer = GPT2Tokenizer.from_pretrained(PATHS['phrase_generator'])
            self.phrases_model = GPT2LMHeadModel.from_pretrained(PATHS['phrase_generator'], pad_token_id=self.phrases_tokenizer.eos_token_id) 
            print('Phrase list generator loaded')

            # Load stop generator model
            self.stop_tokenizer = GPT2Tokenizer.from_pretrained(PATHS['stop_generator'])
            self.stop_model = GPT2LMHeadModel.from_pretrained(PATHS['stop_generator'])
            print('Stop generator loaded')

        else:    # Load baseline stop generator
            self.stop_tokenizer = GPT2Tokenizer.from_pretrained(PATHS['stop_baseline'])
            self.stop_model = GPT2LMHeadModel.from_pretrained(PATHS['stop_baseline'], pad_token_id=self.stop_tokenizer.eos_token_id)
            print('Baseline stop generator loaded')

        ### Infilling model

        # Load BERT classifier (necessary for infill_generator, a2)
        self.infill_classifier = pipeline('text-classification', model=PATHS['position_classifier'])
        print('Infilling classifier loaded')

        # Load GPT-2 infiller    
        if (self.infilling_model_type == 'infill_generator') or (self.infilling_model_type == 'a2_2'):
            self.infill_tokenizer = GPT2Tokenizer.from_pretrained(PATHS['infill_generator'])
            self.infiller_generator = GPT2LMHeadModel.from_pretrained(PATHS['infill_generator'], pad_token_id=self.stop_tokenizer.eos_token_id)
            print('Infill generator loaded')

        elif self.infilling_model_type == 'a2_1':
            self.infill_tokenizer = GPT2Tokenizer.from_pretrained(PATHS['a2_1'])
            self.infiller_generator = GPT2LMHeadModel.from_pretrained(PATHS['a2_1'], pad_token_id=self.stop_tokenizer.eos_token_id)
            print('Infill generator (a2_1) loaded')

        elif self.infilling_model_type == 'a2_3':
            self.infill_tokenizer = GPT2Tokenizer.from_pretrained(PATHS['a2_3'])
            self.infiller_generator = GPT2LMHeadModel.from_pretrained(PATHS['a2_3'], pad_token_id=self.stop_tokenizer.eos_token_id)
            print('Infill generator (a2_3) loaded')

        print('\nAll models successfully loaded.')


    ####################

    def get_stop(self, start):
        ''' Method to generate stop sentence (from phraselist or baseline model)
            Input:  (str) start sentence
            Output: (str) stop sentence
        '''
        
        if self.stop_model_type == 'phrase2stop_gpt':

            # Generate phrases list (given start sentence)
            start_and_rep = start + ' ' + self.get_rep_phrase(start)

            # Generate stop sentence (given start + repetitive phrases)
            # print('Generating stop with phrase list')
            encoding = self.stop_tokenizer.encode(start_and_rep, return_tensors='pt')
            output = self.stop_model.generate(encoding, max_length=150, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
            decoded_output = self.stop_tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract stop (from 'start + rep phrases + stop + extra')
            stop = re.search('\].*', decoded_output)
            if stop is not None:
                stop = stop[0][2:]          # remove extra punctuation

                # if self.interactive:
                #     print(f'Suggested stop: {stop}')
                #     change = input('Would you like to change the stop?(y/n) ')
                #     if change=='y':
                #         stop = input('Please enter your stop: ')
                return stop
            print('ERROR: Stop not generated')     # model did not produce expected output 
            return ''
        
        else:   # Use GPT-2 baseline
            # print('Generating stop with baseline stop model')

            encoding = self.stop_tokenizer.encode(start, return_tensors='pt')
            output = self.stop_model.generate(encoding, max_length=50, num_beams=5, no_repeat_ngram_size=2)
            decoded_output = self.stop_tokenizer.decode(output[0], skip_special_tokens=True)   
            print(decoded_output) 
            # decoded_output = re.search('{.?!}.*', decoded_output)  # extract stop sentence
            decoded_output = sent_tokenize(decoded_output)
            if len(decoded_output) > 1:
                return decoded_output[-1]
            return ''


    def get_rep_phrase(self, start):
        ''' Helper method to get_stop() method
            Generate repetitive phrases from start sentence

            INPUT:  start sentence string
            OUTPUT: repetitive phrases list as string
        '''
        
        # Generate repetitive phrases from start sentence
        # print('Generating phrase list with phrase generator')
        encoding = self.phrases_tokenizer.encode(start, return_tensors='pt')
        output = self.phrases_model.generate(encoding, max_length=50, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
        decoded_output = self.phrases_tokenizer.decode(output[0], skip_special_tokens=True)
        phrases_str = re.search('\[.*\]', decoded_output)    # extract rep phrases

        if phrases_str is not None:
            self.phrases_str = phrases_str[0]

            # Controllability feature --> edit stop sentence
            if self.interactive:
                print(f'Given start: {start} \nSuggested rep phrase: {phrases_str}')
                change = input('Would you like to make any changes?(y/n) ')
                if change != None:
                    self.phrases_str = input('Please enter your rep phrases [...]: ')
            
            return self.phrases_str
        
        else:   
            print('ERROR: Generated phrase list not found')
            return ''
        

    def get_infill_index(self):
        ''' Method to determine location of next infilled sentence
            OUTPUT: (int) index location of most probably next infill location
        '''
        # Prepare current sentences for infilling
        possible_indices = [i for i in range(1, len(self.sentences))]   # list of all indices between endpoints

        # Get list of infill probabilities
        prob_dict, prob_scores = [], []
        for i in range(len(possible_indices)):
            index = possible_indices[i]
            lc = ' '.join(self.sentences[:index])
            rc = ' '.join(self.sentences[index:])
            prob_dict.append(self.infill_classifier(lc + ' <mask> ' + rc)[0])    # Get probability 
            prob_scores.append(prob_dict[i]['score'])

        max_prob_index = prob_scores.index(max(prob_scores))        # position for infill
        if prob_dict[max_prob_index]['label'] == 'NEGATIVE':        # no positive predictions
            max_prob_index = prob_scores.index(min(prob_scores))    # get most likely from negative predictions
        
        return possible_indices[max_prob_index]
    
        

    def get_infill(self, m_input):
        ''' Method to generate infilled sentences
            INPUT:  (str) input to model
            OUTPUT: (str) infill sentence
        '''
        
        # print('Generating infill sentence with GPT-2 infiller_generator')
        encoding = self.infill_tokenizer.encode(m_input, return_tensors='pt')
        output = self.infiller_generator.generate(encoding, max_length=256, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        decoded = self.infill_tokenizer.decode(output[0], skip_special_tokens=True)
        infill = re.search('<SEP>.*', decoded)  # . is any char except \n
        if infill is not None:
            return infill[0][6:]
        else:
            print(f'Unexpected output: {decoded}')
            return ''
        
        
    def build_story(self, start, num_s=3):
        ''' Input:  (str) start sentence
                    (int) number of infill sentences
            Output: (str) generated story
        '''
        
        self.reset()    # Clear cache from previous generations
        
        ### 1. Generate stop (given start)
        stop = self.get_stop(start)

        # Store endpoints
        self.sentences.append(start)                
        self.sentences.append(stop) 
        self.sentences_with_order.append(start)
        self.sentences_with_order.append(stop)    


        #### 2. Infill middle sentences
        if self.infilling_model_type == 'a2_1':
            input_a21 = start + ' ' + stop + ' <MASK> '
            encoded_input = self.infill_tokenizer.encode(input_a21, return_tensors='pt')
            output = self.infiller_generator.generate(encoded_input, max_length=256, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
            output = self.infill_tokenizer.decode(output[0], skip_special_tokens=True)

            # Remove mask and put sentences in order
            output = output.replace('<MASK>', '')   # Remove <MASK> token
            output = sent_tokenize(output)
            stop = output.pop(1)
            output.append(stop)
            output = ' '.join(output)
            return output, len(self.sentences), None, None
            

        for n in range(num_s):

            if self.infilling_model_type == 'a2_3':
                # infill_index = random.randint(1, len(self.sentences)-1)

                input_a23 = ' '.join(self.sentences) + ' <SEP>'
                encoded_input = self.infill_tokenizer.encode(input_a23, return_tensors='pt')
                output = self.infiller_generator.generate(encoded_input, max_length=256, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
                output = self.infill_tokenizer.decode(output[0], skip_special_tokens=True)
                output = output.replace('<SEP>', '')   # Remove <MASK> token
                output = sent_tokenize(output)
                infill = output.pop()
                index = self.get_infill_index()
                self.sentences.insert(index, infill)

            else:
                if self.infilling_model_type == 'a2_2':
                    infill_index = random.randint(1, len(self.sentences)-1)  
                else: 
                    # Get index with maximum probability of needing infill sentence
                    infill_index = self.get_infill_index()
            
                # Prepare left and right contexts
                lc = ' '.join(self.sentences[:infill_index])   # All sentences up to infill position
                rc = ' '.join(self.sentences[infill_index:])   # All sentences after infill (include current sentence in position)
                
                # Generate single infill sentence
                m_input = lc + ' <INFILL_LOC> ' + rc + ' <SEP>'
                infill = self.get_infill(m_input)

                if infill != '':
                    self.sentences.insert(infill_index, infill)
                    self.sentences_with_order.insert(infill_index, '[' + str(n+1) + '] ' + infill)

        # return (string of concatenated sentences, total number of sentences, string of numbered sentences)
        return ' '.join(self.sentences), len(self.sentences), ' '.join(self.sentences_with_order), self.phrases_str
    
    def reset(self):
        '''Clear settings for next generation with same model'''
        self.sentences = []
        self.sentences_with_order = []
        self.phrases_str = ''