import re

def flatten(lst):
    return [item for sublst in lst for item in sublst]
 
    
def clear(context):
    res = []
    for line in context:
        line = re.sub(r'[^\w\s]','', line).lower()
        res.append(line)
    return res


def label_function(line, label):
    res = []
    for i in line.split():
        res.append('I-'+ label)
    res[0] = res[0].replace('I-', 'B-')
    res = ' '.join(res)
    return res


def choose_samples(single_hop_dataset, labels):
    chosen_samples = []
    seq_in_res = single_hop_dataset[0]
    seq_out_res = single_hop_dataset[1]
    label_res = single_hop_dataset[2]
    
    for k, i in enumerate(label_res):
        if i in labels:
            chosen_samples.append([seq_in_res[k], seq_out_res[k], label_res[k]])
    return chosen_samples


def find_slot_name(seq_out):
    seq_out = set(seq_out.replace('O', '').replace('B-', '').replace('I-', '').split())
    return seq_out


def find_slot_index(seq_in, seq_out):
    for el in range(len(seq_out.split())):
        try:                  
            idx = seq_out.split().index(seq_out.replace('O', '').split()[0])
            in_split = seq_in.split()
            out_split = seq_out.split()
            del in_split[idx]
            del out_split[idx]
            seq_in = ' '.join(in_split)
            seq_out = ' '.join(out_split)
        except:
            continue
    return [seq_in, seq_out]


def find_intent(i):
    return re.findall('\<([^>]+)', i)


def find_slot(i):
    return re.findall('\[([^]]+)', i)


def find_slot_content(i):
    return re.findall('(\S+):\w-', i)


def find_slot_label(i):
    return re.findall(':\w-(\S+)', i)


def find_param(i):
    return re.findall('(\?\w+)', i)