import re
import numpy as np
from additional_functions import flatten, clear
from russian_paraphrasers import Mt5Paraphraser
paraphraser_mt5 = Mt5Paraphraser(model_name="mt5-base", range_cand=False, make_eval=False)


def generation(intent_label, intent_label_translation, num_examples_per_intent):
    """ 
    Context generation for further dataset collection.
    
    Arguments:
    intent_label -- a list with knowledge graph relations
    intent_label_translation -- a list with translated knowledge graph relations
    num_examples_per_intent -- a list with the required number of generated samples per relation
    
    Return:
    context_processed -- a list with generated context
    context_label_processed -- a list with corresponded relations labels
    
    """
    generated_context = []
    for i, j in zip(intent_label_translation, num_examples_per_intent):
        i = i.split(', ')
        generated_context_one_relation = []
        for k in i:
            num = int(np.ceil(j/len(i)))
            results = paraphraser_mt5.generate(k, n=num, temperature=5.5, top_k=20, top_p=0.3, max_length=100, repetition_penalty=2.5, threshold=0.5)
            generated_context_one_relation.append(clear(results['results'][0]['predictions']))   
    
        generated_context.append(generated_context_one_relation)  
    
    context_processed = []
    context_label_processed = []
    for i, j, k in zip(generated_context, num_examples_per_intent, intent_label):
        i = flatten(i)
        if len(i) >= j:
            i = i[:j]
        else:
            i = i + i[:j - len(i)]
        context_processed.append(i)
        k = [k] * j
        context_label_processed.append(k)
    
    return [context_processed, context_label_processed]

