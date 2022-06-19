import re
import itertools
from additional_functions import flatten, find_intent, find_slot, find_slot_content, find_slot_label, find_param
from owlready2 import *
onto = get_ontology("ontology_name.owl").load()


def intent_template_function(intent_labels):
    """ Create a intents template for further SPARQL query. """
    
    intent_template = {}
    for k in intent_labels:
        intent_template[k] = ''
    return intent_template


def slot_template_function(slot_labels):
    """ Create a slots template for further SPARQL query. """
    
    slot_template = {}
    for k in slot_labels:
        if '-' in k:
            if k[2:] not in slot_template:
                slot_template[k[2:]] = ''
    return slot_template

                
def fill_templates(intent_labels, slot_labels, JointBERT_res): 
    """
    Fill created templates with JointBERT detection results.
    
    Arguments:
    intent_labels -- a list with intent labels (knowledge graph relations)
    slot_labels -- a list with slots labels (knowledge graph entities)
    JointBERT_res -- a list with JointBERT detection results
    
    Return:
    intent_res_list -- a template filled with detected relations
    slot_res_list -- a template filled with detected entities

    """
    intent_res_list = []
    slot_res_list = []

    for i in JointBERT_res:
        intent_template = intent_template_function(intent_labels)
        slot_template = slot_template_function(slot_labels)

        # find intents (e.g. ['has_brand']) and add to template (e.g. 'has_brand': 'has_brand')
        intent = find_intent(i)
        for k in intent[0].split('#'):
            intent_template[k] = k


        # find slots (e.g. ['Figaro:B-brand']), content (e.g. Figaro) and label (e.g. brand) 
        # and add to template (e.g. 'brand': 'Figaro')
        slot = find_slot(i)
        slot_content = list(map(find_slot_content, slot))
        slot_label = list(map(find_slot_label, slot))

        slot_content_pairs = []
        for k,j in zip(slot_label, slot_content):
            slot_content_pairs.append(flatten([k, j]))

        for k in slot_content_pairs:
            if slot_template[k[0]] == '':
                slot_template[k[0]] = k[1]
            else:
                slot_template[k[0]] = ' '.join([slot_template[k[0]], k[1]])

        intent_res_list.append(intent_template)
        slot_res_list.append(slot_template)
    return [intent_res_list, slot_res_list]


def intent_to_SPARQL(intent_labels, ontology_name):
    """ Create a dictionary with knowledge graph relations and corresponded triplets pattern for further SPARQL generation. """
    
    intent_to_SPARQL_triples = {}
    for i in intent_labels:
        query = list(default_world.sparql(" SELECT * { ?q %s:%s ?w . } LIMIT 1" % (ontology_name, i)))[0]
        sbj = type(query[0]).name.lower()
        obj = str(i).replace('has_', '')   
        intent_to_SPARQL_triples[i] = '?%(sbj)s %(ontology_name)s:%(pred)s ?%(obj)s .' % {'sbj': sbj, 'ontology_name': ontology_name, 'pred': i, 'obj': obj}
    return intent_to_SPARQL_triples


def slot_to_data_type(intent_labels, slot_labels, ontology_name):
    """ Create a dictionary with knowledge graph entities and corresponded literals data types. """
    
    slot_to_type = slot_template_function(slot_labels)
    for i in intent_labels:
        query = list(default_world.sparql(" SELECT * { ?q %s:%s ?w . } LIMIT 1" % (ontology_name, i)))[0]
        obj = str(i).replace('has_', '') 
        data_type = type(query[1])
        slot_to_type[obj] = data_type
    return slot_to_type


def queries_generation(intent_labels, slot_labels, intent_res_list, slot_res_list, class_instances_name, ontology_name):
    """
    SPARQL queries generation for single-hop and multi-hop question-answering based on JointBERT detection results.
    
    Arguments:
    intent_labels -- a list with intent labels (knowledge graph relations)
    slot_labels -- a list with slots labels (knowledge graph entities)
    intent_res_list -- a template filled with detected relations
    slot_res_list -- a template filled with detected entities
    class_instances_name -- a list with knowledge graph entities related to class instances
    ontology_name -- the name of ontology
    
    Return:
    SPARQL_strings -- a list with generated SPARQL queries
    
    """
    SPARQL_strings = []
    SPARQL_param = []
    
    # create a dictionaries with knowladge graph triplets pattern and literals data types
    intent_to_SPARQL_triples = intent_to_SPARQL(intent_labels, ontology_name)
    slot_to_type = slot_to_data_type(intent_labels, slot_labels, ontology_name)
    
    # generate SPARQL template
    for k in range(len(intent_res_list)):
        start = 'SELECT * {'
        end = ' }'
        for i in intent_res_list[k]:
            if i == intent_res_list[k][i]:
                start = ' '.join([start, intent_to_SPARQL_triples[i]])
        SPARQL_string = start + end

        # fill SPARQL template based on literals data types
        for i in slot_res_list[k]:
            if slot_res_list[k][i] != '':
                if i in class_instances_name:
                    SPARQL_string = SPARQL_string.replace('?' + i, '%s:' % ontology_name + slot_res_list[k][i].replace(' ', '_'))
                elif slot_to_type[i] == float:
                    SPARQL_string = SPARQL_string.replace('?' + i, '"' + slot_res_list[k][i] + '"' + '^^xsd:decimal')
                elif slot_to_type[i] == int:
                    SPARQL_string = SPARQL_string.replace('?' + i, '"' + slot_res_list[k][i] + '"' + '^^xsd:integer')
                elif slot_to_type[i] == str:
                    SPARQL_string = SPARQL_string.replace('?' + i, '"' + slot_res_list[k][i] + '"' + '^^xsd:string')

        param = set(find_param(SPARQL_string))
        SPARQL_string = SPARQL_string.replace('*', ' '.join(param))
        
        # SPARQL post-processing for multi-hop question-answering
        if len(param) > 1:
            for i in list(itertools.combinations(param, 2)):
                for j in intent_to_SPARQL_triples.values():
                    if i[0]+' ' in j and i[1]+' ' in j and j not in SPARQL_string:
                        SPARQL_string = SPARQL_string.replace('}', ' '.join([j, ' }']))
        
        SPARQL_param.append(param)
        SPARQL_strings.append(SPARQL_string.replace('–', '-'))
    return SPARQL_strings

