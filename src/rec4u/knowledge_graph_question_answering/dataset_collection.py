import random
from additional_functions import flatten, label_function, choose_samples, find_slot_name, find_slot_index
from owlready2 import *
onto = get_ontology('/home/jovyan/notebooks/_Main/Car4U/Ontology/autobot.owl').load()


def collect_dataset_single_hop(intent_label, intent_label_translation, num_examples_per_intent, generated_context, ontology_name):
    
    """ 
    Dataset collection for single-hop knowledge graph question-answering.
    
    Arguments:
    intent_label -- a list with knowledge graph relations
    intent_label_translation -- a list with translated knowledge graph relations
    num_examples_per_intent -- a list with the required number of generated samples per relation
    generated_context -- a list with context generated based on provided knowledge graph relations
    ontology_name -- the name of ontology
    
    Return:
    single_hop_seq_in -- a list with training samples
    single_hop_seq_out -- a list with slot (knowledge graph entities) labels
    single_hop_labels -- a list with intent (knowledge graph relations) labels
    
    """
    seq_in_res = []
    seq_out_res = []
    label_res = []
    for i, j in zip(intent_label, num_examples_per_intent):
        seq_in = []
        seq_out = []
        line_label = []
        query = list(default_world.sparql(" SELECT * { ?q %s:%s ?w . } " % (ontology_name, i)))[:j]
        for k in query:
            sbj_line = k[0].name.replace('_', ' ')

            if ontology_name in str(k[1]):
                obj_line = k[1].name.replace('_', ' ') # object is a class instance
                selected_label = label_function(obj_line, str(i).replace('has_', ''))
                selected_line = obj_line 
            else:
                obj_line = str(k[1]) # object is a literal
                if query.index(k) % 2 == 0:
                    selected_label = label_function(sbj_line, type(k[0]).name.lower())
                    selected_line = sbj_line
                else:
                    selected_label = label_function(obj_line, str(i).replace('has_', ''))
                    selected_line = obj_line 

            seq_in.append(selected_line)
            seq_out.append(selected_label)
            line_label.append(i)

        seq_in_res.append(seq_in)
        seq_out_res.append(seq_out)
        label_res.append(line_label)
        
    seq_in_res, seq_out_res, context_res, single_hop_labels = flatten(seq_in_res), flatten(seq_out_res), flatten(generated_context), flatten(label_res)

    single_hop_seq_in = []
    single_hop_seq_out = []
    for i, o, c in zip(seq_in_res, seq_out_res, context_res):
        context = c.split()
        context_len = len(c.split())
        context_label = ['O']*context_len
        num = random.randint(0, context_len)

        if num != context_len:
            context.insert(num, i) 
            context_label.insert(num, o)

            single_hop_seq_in.append(' '.join(context))
            single_hop_seq_out.append(' '.join(context_label))

        else:
            context.append(i) 
            context_label.append(o)

            single_hop_seq_in.append(' '.join(context))
            single_hop_seq_out.append(' '.join(context_label))
    
    return [single_hop_seq_in, single_hop_seq_out, single_hop_labels]


def collect_dataset_multi_hop(single_hop_dataset, chosen_label):
    """ 
    Dataset collection for multi-hop knowledge graph question-answering.
    
    Arguments:
    single_hop_dataset -- resultes of single-hop question-answering dataset collection
    chosen_label -- a list of labels chosen for multi-hop question-answering
    
    Return:
    multi_hop_seq_in -- a list with training samples
    multi_hop_seq_out -- a list with slot (knowledge graph entities) labels
    multi_hop_label -- a list with intent (knowledge graph relations) labels
    
    """
    multi_hop_seq_in = []
    multi_hop_seq_out = []
    multi_hop_labels = []
    chosen_samples = choose_samples(single_hop_dataset, chosen_label)
    for i in chosen_samples:
        for j in chosen_samples:
            first_out = find_slot_name(i[1])
            second_out = find_slot_name(j[1])
            if i[2] != j[2]:
                if first_out != second_out:
                    multi_hop_seq_in.append(' '.join([i[0], j[0]])) 
                    multi_hop_seq_out.append(' '.join([i[1], j[1]])) 
                    multi_hop_labels.append('#'.join([i[2], j[2]]))
                else:
                    threshold = random.uniform(0, 1)
                    if threshold > 0.5:
                        i[0], i[1] = find_slot_index(i[0], i[1])
                        multi_hop_seq_in.append(' '.join([i[0], j[0]])) 
                        multi_hop_seq_out.append(' '.join([i[1], j[1]])) 
                        multi_hop_labels.append('#'.join([i[2], j[2]]))
                    else:
                        j[0], j[1] = find_slot_index(j[0], j[1])
                        multi_hop_seq_in.append(' '.join([i[0], j[0]])) 
                        multi_hop_seq_out.append(' '.join([i[1], j[1]])) 
                        multi_hop_labels.append('#'.join([i[2], j[2]]))
                del chosen_samples[chosen_samples.index(i)]
                del chosen_samples[chosen_samples.index(j)]
                break
    return [multi_hop_seq_in, multi_hop_seq_out, multi_hop_labels]
            
    
