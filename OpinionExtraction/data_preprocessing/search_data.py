import os
import re
import nltk
import copy
from collections import defaultdict, OrderedDict
import numpy as np

docsdir = 'database.mpqa.2.0/docs'
man_ann_dir ='database.mpqa.2.0/man_anns'
man_ann_file = 'gateman.mpqa.lre.2.0'


###############get the files from man_ann##########################
def walk_ann_dir():
    filenames = []
    for rootdir, subdir, dirs in os.walk(man_ann_dir):
        for dir in dirs:
            dirpath=os.path.join(rootdir,dir)
            if 'gateman' in dirpath: # only need ann file
                filenamel=dirpath.replace('\\','/').split('/')
                filenamel='/'.join(filenamel[-3:-1])
                filenames.append(filenamel)
    return filenames

#generate index from a span,then get the text content from this index.
def sub(span,text):
    """return a snippet of a given text
    >>> sub((5,8), "I love munich even if it is crowded.")
    'e m'
    """
    index_start=span[0]
    index_end=span[1]
    return text[index_start:index_end]

def bigger_span(span1,span2):
    '''
    show true if span1 is bigger than span2
    >>> bigger_span((225,229),(229,240))
    False
    >>> bigger_span((228,290),(230,239))
    True
    '''
    if (span1[1]-span1[0]) > (span2[1]-span2[0]):
        return True
    else: return False

##### if a span is within another span.
def is_within_(small_span,bigger_span):
    """return true if smaller span is within the bigger span
    >>> is_within_((730,830),(720,900))
    True
    >>> is_within_((710,830),(720,900))
    False
    >>> is_within_((730,910),(730,911))
    True
    """
    if small_span[0]>= bigger_span[0] and small_span[1]<=bigger_span[1]:
        return True
    else: return False

def overlapping(span1, span2):
    """
    check if it is real overlapping relation without being sub relation
    >>> overlapping((225,238),(225,257))
    False
    >>> overlapping((225,238),(226,231))
    False
    >>> overlapping((225,238), (226,241))
    True
    """
    if span1[0]<span2[0] and span1[1]>span2[0] and span1[1]< span2[1]:
        return True
    elif span2[0]<span1[0] and span2[1]>span1[0] and span2[1]< span1[1]:
        return True
    else : return False

##################### get the token span dictionary from the sentence span and text ####################################
def get_token_span_dict_from_sent_span(sent_span, text):
    """ get token span dict from sent_span and its text
    >>> get_token_span_dict_from_sent_span((1,10)," I love munich even")
    {0: {(1, 2): 'I'}, 1: {(3, 7): 'love'}, 2: {(8, 10): 'mu'}}
    """
    sent = sub(sent_span, text)
    tokens = nltk.word_tokenize(sent)
    tokens = ['"' if tok in ['``', '"', '\'\''] else tok for tok in tokens]  ###very important patch
    token_span_dict = dict()
    token_offset = sent_span[0]  ####very important offset setting
    for token_id, token in enumerate(tokens):
        token_span_ = dict()
        token_offset_copy = copy.copy(token_offset)  # patch for value error tokenizer
        token_offset = text.find(token, token_offset)
        if token_offset == -1:
            token_offset = token_offset_copy + 1
        token_span_[(token_offset, token_offset + len(token))] = token

        token_span_dict[token_id] = token_span_
        token_offset += len(token)
    return token_span_dict  ### a token span dict in one sentence
####################################################################################

class Document:
    def __init__(self,docid,text,ann_dict,dse_exist,sentence_data):
        self.docid=docid
        self.text=text
        self.ann_dict=ann_dict
        self.dse_exist=dse_exist
        self.sentence_data=sentence_data

    @classmethod
    def from_file(cls,docid):
        ##############get ann_list for the doc########################
        dse_exist=False
        ann_dict={}
        counter_dse=0
        doc_id_ann = man_ann_dir + '/' + docid + '/' + man_ann_file
        with open(doc_id_ann, 'r') as ann_file:
            for line in ann_file.readlines():
                line = line.replace('\n', '')
                if line[0] == '#':
                    continue
                id_, span_, data_type_, ann_type_, attributes_ = line.split('\t')
                if ann_type_=='GATE_direct-subjective':counter_dse+=1
                #print(ann_type_)
                span_list = list(map(int, span_.split(',')))

                att_dict = dict(list(map(lambda x: (x[0].strip(), x[1].strip()),
                                         re.findall(r"(.+?)=\"(.+?)\"", attributes_.strip()))))
                ann_dict[int(id_)]={
                                    "span" : (span_list[0],span_list[1]),
                                    "data_type" : data_type_,
                                    "ann_type" : ann_type_,
                                    "attributes" : att_dict
                }
            ann_file.close()

        if counter_dse >0: dse_exist=True
        doc_id = docsdir + '/' + docid
        with open(doc_id, 'r', encoding='utf-8') as file:
            byte_list = [byte for byte in [line for line in file.readlines()]]
            text = ''.join(byte_list)

        ###get the sentence spans and ids in each file from gatesentences.##########
        sentence_data={}###{0: (969, 1210), 1: (1474, 1546),...}
        with open(os.path.join(man_ann_dir,docid,"gatesentences.mpqa.2.0")) as f:
            for line in f.readlines():
                line=line.replace('\n','')
                id0,span0,data_type0,ann_type0 = line.split('\t')
                id0=int(id0)
                spans=list(map(int,span0.split(',')))
                sentence_data[id0]=(spans[0],spans[1])

        return cls(docid,text,ann_dict,dse_exist,sentence_data)

    def get_dse_sentences(self):
        if self.dse_exist:
            sent_span_for_dse = {}
            for sent_id, sent_span in self.sentence_data.items():
                for id, d in self.ann_dict.items():
                    if d['ann_type']=='GATE_direct-subjective':
                        if is_within_(d['span'],sent_span):
                            sent_span_for_dse[id]=sent_span
        return sent_span_for_dse###{id_dse:sent_span}

    #incomplete, every dse with its possible holder and target, but ignore nested sources.
    def get_holder_and_target_for_dse(self):
        sent_holder_target_dict=defaultdict(list)
        if self.dse_exist:
            sent_span_for_dse=self.get_dse_sentences()

            for dse_id, sent_span in sorted(sent_span_for_dse.items()):
                holder_target_dict={}

                # give dse a tag 'O'
                d=self.ann_dict[dse_id]
                dse_span=d['span']
                holder_target_dict[dse_span]='O' #{dse_span:'O'}

                source = d['attributes'].get('nested-source', '')
                source_list = list(source.split(','))

                source_list = [source.strip() for source in source_list]

                source_ = ', '.join(source_list)  # nested-source ="w, bush"
                source_0 = ','.join(source_list)  # nested-source="w,bush"

                for id,de in self.ann_dict.items():
                    if (source_  == de['attributes'].get('nested-source','')
                        or source_0 == de['attributes'].get('nested-source','')):
                        if  de['ann_type']=='GATE_agent' and is_within_(de['span'],sent_span):
                            holder_target_dict[de['span']]='H'   #{holder_span:'H'}

                if 'attitude-link' in d['attributes'].keys():
                    at_ = d['attributes'].get('attitude-link', '')  # get the dictionary of attitude links
                    attitude_link_id = re.search(r'a\d+', at_)
                    if attitude_link_id is not None:
                       attitude_link_id = attitude_link_id.group(0) #only get the first attitude link id.
                       for id,de in self.ann_dict.items():
                           if attitude_link_id in de['attributes'].get('id', '') and de['ann_type'] == 'GATE_attitude':
                               target_ = de['attributes'].get('target-link', '')
                               target_link = re.search(r't\d+', target_)
                               if target_link is not None:
                                   target_link_id = target_link.group(0) # only get the first target link id.
                                   for id, dei in self.ann_dict.items():
                                       if target_link_id in dei['attributes'].get('id', '') and dei['ann_type'] == 'GATE_target':
                                           if is_within_(dei['span'], sent_span):
                                               holder_target_dict[dei['span']] = 'T'
                sent_holder_target_dict[sent_span].append(holder_target_dict) #{sent_span:{holder_span:'H',dse_span:'D',target_span:'T'}
        return sent_holder_target_dict

    # exclude overlapping! but include complete and incomplete
    #######check later for priorities.
    def target_holder_dict_no_overlapping(self,target_holder_dict):
        from collections import ChainMap
        from random import choice
        target_holder_dict_no_overlapping={}
        for sent_span, span_tag_dict_list in sorted(target_holder_dict.items()):
            try:
                print('sentence: ', sub(sent_span,self.text))
            except Exception:
                continue
            print('original span tag dict list',span_tag_dict_list)
            if len(span_tag_dict_list)==1:
                target_holder_dict_no_overlapping[sent_span]=span_tag_dict_list
            elif len(span_tag_dict_list)>1:
                #######################################################################
                combined_dict={k:[d.get(k) for d in span_tag_dict_list if d.get(k) is not None] for k in set().union(*span_tag_dict_list)}
                chain_dict=dict(ChainMap(*span_tag_dict_list))
                ########################################################################
                print('if more than one tag dict, combined_dic:' , sorted(combined_dict.items()))
                print(combined_dict)
                print('chained dict: ',chain_dict)
                if any (len(set(tag_list))>1 for tag_list in combined_dict.values()):
                    print('if tag_list_set > 1')
                    temporay_d ={}
                    for span,tag_list in combined_dict.items():
                        if(len(set(tag_list))>1):
                            temporay_d[span] = choice(tag_list)
                        else: temporay_d[span] =tag_list[0]
                    print('if more than one tag :', temporay_d)
                    ##########################################################
                    if any(overlapping(k1, k2) for k1 in sorted(combined_dict) for k2 in sorted(temporay_d) if k1 is not k2):
                        #check if any two just overlapping without subrelations
                        target_holder_dict_no_overlapping[sent_span] = temporay_d
                        print('result ',temporay_d)
                        print('overlapping,no sub \n')
                    elif (any(is_within_(k1, k2) for k1 in sorted(temporay_d) for k2 in sorted(temporay_d) if k1 is not k2)):
                        subrelation_spans =[(k1,k2)  for k1 in sorted(temporay_d) for k2 in sorted(temporay_d) if k1 is not k2 if is_within_(k1,k2) and bigger_span(k2,k1)]
                        print(subrelation_spans)
                        exclude_spans=[]

                        for k1, k2 in subrelation_spans:
                            #if they have the same tags, exclude the smaller one
                            if temporay_d[k1]==temporay_d[k2]:
                                if k1 not in exclude_spans:
                                    exclude_spans.append(k1)
                            else: #else, exclude the bigger one, to retain the subrelation.
                                #if k2 not in exclude_spans:
                                exclude_spans.append(k1)
                        print(exclude_spans)
                        sub_dict={span:chain_dict[span] for span in temporay_d if span not in set(exclude_spans)}
                        target_holder_dict_no_overlapping[sent_span] = sub_dict
                        print('result ',sub_dict)
                        print('subrelations \n')
                    else:
                        print('same with original')
                        print('no sub, no overlapping \n')
                        target_holder_dict_no_overlapping[sent_span]=temporay_d

                elif all(len(set(tag_list)) == 1 for tag_list in combined_dict.values()):
                    #########sub relations possible.############################################
                    print('set of tag_list ==1 ')
                    if any(overlapping(k1, k2) for k1 in sorted(combined_dict) for k2 in sorted(combined_dict) if k1 is not k2):
                        #check if any two just overlapping without subrelations
                        target_holder_dict_no_overlapping[sent_span] = chain_dict
                        print('result ',dict(ChainMap(*span_tag_dict_list)))
                        print('overlapping,no sub \n')
                    elif (any(is_within_(k1, k2) for k1 in sorted(combined_dict) for k2 in sorted(combined_dict) if k1 is not k2)):
                        subrelation_spans =[(k1,k2)  for k1 in sorted(combined_dict) for k2 in sorted(combined_dict) if k1 is not k2 if is_within_(k1,k2) and bigger_span(k2,k1)]
                        exclude_spans=[]

                        for k1,k2 in subrelation_spans:
                            #if they have the same tags, exclude the smaller one
                            if chain_dict[k1]==chain_dict[k2]:
                                if k1 not in exclude_spans:
                                    exclude_spans.append(k1)
                            else: #else, exclude the bigger one, to retain the subrelation.
                                exclude_spans.append(k1)
                        print(exclude_spans)
                        sub_dict={span:chain_dict[span] for span in chain_dict if span not in set(exclude_spans)}
                        target_holder_dict_no_overlapping[sent_span] = sub_dict
                        print('result ',sub_dict)
                        print('subrelations \n')
                    else:
                        print('same with original')
                        print('no sub, no overlapping \n')
                        target_holder_dict_no_overlapping[sent_span]=span_tag_dict_list
                        # print('if overlapping with sub relation-->',d)
        # print('result -->',target_holder_dict_no_overlapping)
        return target_holder_dict_no_overlapping

    def target_holder_dict_for_doc(self,sent_holder_target_dict):
        sent_tag_dict={}
        for sent_span,tags in sorted(sent_holder_target_dict.items()):
            token_span_dict = get_token_span_dict_from_sent_span(sent_span,self.text) #{token_id: {token_span:toke}}
            # tags = sent_holder_target_dict[sent_span] # {token_span:'H',target_span:'T'...} , might be list if not nonoverlapping.
            token_id_span_tag_dict={}
            for token_id, token_span_ in sorted(token_span_dict.items()):
                token_span_tag_dict={}
                for token_span, token in sorted(token_span_.items()):
                    #tag_id_dict={}
                    #token_span_tag_dict[token_span]=tag_id_dict
                    if type(tags) is list:
                        for tag_dict_id, tag_dict in enumerate(tags): #tag_dict_id for the 1st or 2nd holder/target.
                            for tag_span, tag in sorted(tag_dict.items()):
                                if is_within_(token_span,tag_span):
                                   if token_span[0]==tag_span[0]:
                                       token_span_tag_dict[token_span]='B_'+tag
                                   else:
                                       token_span_tag_dict[token_span]='I_'+tag
                    elif type(tags) is dict:
                        for tag_span, tag in sorted(tags.items()):
                            if is_within_(token_span, tag_span):
                                if token_span[0] == tag_span[0]:
                                    token_span_tag_dict[token_span] = 'B_' + tag
                                else:
                                    token_span_tag_dict[token_span] = 'I_' + tag
                token_id_span_tag_dict[token_id]=token_span_tag_dict   #{token_id:{token_span: token_tag}..}
            sent_tag_dict[sent_span]=token_id_span_tag_dict #{sent_span: {token_id:{token_span:token_tag}....}...}
        return sent_tag_dict

    # sent_tag_dict could be incomplete or complete .
    # result : overlapping possible, tag_id need to be deleted, nested_holder=False
    def get_data(self,sent_tag_dict):# sent_tag_dict could
        sent_id_token_tag_dict=dict()#{id:{token:tag}} finals. until here, all correct and deterministic
        dse_sentences=[]
        for sent_id, sent_span1 in sorted(self.sentence_data.items()):
            for sent_span,token_id_span_tag_dict in sorted(sent_tag_dict.items()):
                id_token_tag_dict=dict()
                #print('sent_span:',sent_span)
                token_span_dict=get_token_span_dict_from_sent_span(sent_span, self.text)
                #print('token_span_dict: ',token_span_dict)
                #print('token_id_span_Tag_dict: ',token_id_span_tag_dict)
                for token_id, token_spans in sorted(token_span_dict.items()):
                    token_span_tag_dict=token_id_span_tag_dict[token_id] #{token_span:{tag_id:tag}} or {}.
                    for token_span, token in sorted(token_spans.items()):
                        if not token_span_tag_dict:
                            id_token_tag_dict[token_id] = (token, 'O')
                        else: id_token_tag_dict[token_id] = (token, token_span_tag_dict[token_span])
                if sent_span1==sent_span:
                    sent_id_token_tag_dict[sent_id]=id_token_tag_dict
                    dse_sentences.append(sent_span)
        return sent_id_token_tag_dict, dse_sentences

    def get_nonoverlap_incomplete(self):
        incomplete_sent_tag_dict=self.get_holder_and_target_for_dse()
        nonoverlap_incomplete=self.target_holder_dict_no_overlapping(incomplete_sent_tag_dict)
        sent_tag_dict=self.target_holder_dict_for_doc(nonoverlap_incomplete)
        data,x=self.get_data(sent_tag_dict)
        return data

if __name__=="__main__":
    import doctest
    doctest.testmod()

    import json,sys

    files = walk_ann_dir()
    dse_sentences_dict=dict()
    print(files)

    sys.stdout = open('data/dse_process.txt','w')



    nonoverlap_incomplete_list=[]
    for file in files:
        print('\nfilename *---->*', file)
        doc = Document.from_file(file)

        de = doc.get_nonoverlap_incomplete()
        if doc.dse_exist:
            if not de == {}:
                nonoverlap_incomplete_list.append(de)

    print(len(nonoverlap_incomplete_list))

    with open('data/dse_.json','w') as fp:
         json.dump( nonoverlap_incomplete_list, fp )