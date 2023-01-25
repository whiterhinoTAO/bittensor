from transformers import StoppingCriteriaList
import bittensor as bt

tokenizer = bt.tokenizer()

stop_words = ['Human:', 'Humans:', 'human:', 'humans:']
stop_word_ids = [tokenizer.encode(stop_word) for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList(stop_word_ids)

