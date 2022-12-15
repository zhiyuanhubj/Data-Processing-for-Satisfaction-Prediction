from transformers import AdamW, BertTokenizer, BertModel
import copy
import torch
import numpy as np
import os
import pickle
import csv
from textaugment import EDA, Translate, Wordnet
from random import randint, sample
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_main_score(scores):
    number = [0, 0, 0, 0, 0]
    for item in scores:
        number[item] += 1
    score = np.argmax(number)
    return score


def load_data(dirname, tokenizer):
    print(dirname)

    name = 'hierarchical_data'

    #if os.path.exists(f'{dirname}-{name}.pkl'):
    #    return read_pkl(f'{dirname}-{name}.pkl')


    raw = [line[:-1] for line in open(dirname, encoding='utf-8')]
    data = []
    for line in raw:
        if line == '':
            data.append([])
        else:
            data[-1].append(line)
    x = []
    emo = []
    act = []
    action_list = {}
    for session in data:
        his_input_ids = []
        for turn in session:
            role, text, action, score = turn.split('\t')
            score = score.split(',')
            action = action.split(',')
            action = action[0]
            if role == 'USER':
                x.append(copy.deepcopy(his_input_ids))
                emo.append(get_main_score([int(item) - 1 for item in score]))
                action = action.strip()
                if action not in action_list:
                    action_list[action] = len(action_list)
                act.append(action_list[action])

            ids = tokenizer.encode(text.strip())[1:]
            his_input_ids.append(ids)


    action_num = len(action_list)

    action_list['None'] = action_list.pop('')

    data = [x, emo, act, action_num,action_list]
    write_pkl(data, f'{dirname}-{name}.pkl')
    return data


def augment_text(List):
    """
    Available methods (randomly chosen):
        1. random_deltion
        2. random_swap
        3. random_insertion
        4. synonym_replacement (wordnet based)
        5. back translation
    """
    eda = EDA()
    result = []
    for i in List:
        if i[0] == "satisfaction score: ":
            continue
        if i[0][:3] == "sat" and i[1] != 3:
            result.append(i)
    print('!!!!!!!!!')
    print(len(result))
    print(result[0][0][:20])
    #print(result[0][0][20:])

    new_result = []
    upsample_factors = 9
    static = [0,0,0,0,0]
    fail = [0,0,0,0,0]
    for i in result:
        text = i[0][20:]

        for j in range(upsample_factors):
            rand_num = randint(1, 5)

            static[rand_num-1] += 1
            try:
                if rand_num == 1:
                    
                    augmented_text = eda.random_deletion(text, p=0.2)
                elif rand_num == 2:
                    augmented_text = eda.random_swap(text,n=1 if int(len(text.split())*0.05) == 0 else int(len(text.split())*0.05))
                elif rand_num == 3:
                    augmented_text = eda.random_insertion(text,n=1 if int(len(text.split())*0.05) == 0 else int(len(text.split())*0.05))
                elif rand_num == 4:
                    augmented_text = eda.synonym_replacement(text,n=1 if int(len(text.split())*0.1) == 0 else int(len(text.split())*0.1))
                else:
                    target_lang = sample(['ko', 'it', 'fa', 'es', 'el', 'la'], k=1)[0]
                    augmented_text = Translate(src='en', to=target_lang).augment(text)   
                    #print('yes')             
                new_result.append(["satisfaction score: "+augmented_text,i[1]])
            except:

                fail[rand_num-1] += 1
                continue

    return new_result




def data_processing(fold=0, data_name='MWOZ'):
    print('[TRAIN]')

    data_name = data_name.replace('\r', '')

    print('dialog used', dialog_used)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # x, emo, act1, act2, action_num1, action_num2 = load_ccpe(f'dataset/{data_name}', tokenizer)

    ## load the previous data
    ## x 是原文，emo是满意度得分，act是这句话的act表征？action_num是什么？
    
    x, emo, act, action_num, action_list = load_data(f'dataset/{data_name}.txt', tokenizer)

    #y = tokenizer.decode(x[2][1])
    keys = []
    for act_element in act:
        key = [k for k,v in action_list.items() if v == act_element]
        keys.append(key[0])



    ## -- devided into ten-fold dataset
    ll = int(len(x) / 10)
    train_x = x[:ll * fold] + x[ll * (fold + 1):]
    train_sat = emo[:ll * fold] + emo[ll * (fold + 1):]
    train_act = keys[:ll * fold] + keys[ll * (fold + 1):]


    test_x = x[ll * fold:ll * (fold + 1)]
    test_sat = emo[ll * fold:ll * (fold + 1)]
    test_act = keys[ll * fold:ll * (fold + 1)]

    
    ## -- list for saving data
    save_result_train = []
    save_result_test = []
    save_result_train_temp = []
    save_result_test_temp = []

    ## generate the data for training. transfer the idx into token
    for j in range(len(train_x)):
        tempt = ''
        if len(train_x[j]) == 0:
            save_result_train_temp.append(tempt)
            continue 
        for k in train_x[j]:
            tempt += tokenizer.decode(k)+' '
        save_result_train_temp.append(tempt)

    for j in range(len(test_x)):
        tempt = ''
        if len(test_x[j]) == 0:
            save_result_test_temp.append(tempt)
            continue
        for k in test_x[j]:
            tempt += tokenizer.decode(k)+' '
        save_result_test_temp.append(tempt)


    ## train set
    for j in range(len(train_x)):
        
        tempt = ''
        if len(train_x[j]) == 0:
            save_result_train.append(['satisfaction score: '+ tempt,train_sat[j]+1])
            # Action Prediction Data
            save_result_train.append(['action prediction: ' + tempt,train_act[j]])
            #save_result_train.append(['utterance generation: '+tempt,''])
            continue 
        for k in train_x[j]:
            tempt += tokenizer.decode(k)+' '

        save_result_train.append(['satisfaction score: '+ tempt,train_sat[j]+1])
        # Action Prediction data
        save_result_train.append(['action prediction: '+ tempt,train_act[j]])
        # Utterance Generation data
        
        if j+1 == len(train_x):
            continue
        if len(train_x[j+1]) == 0:
            continue
        else:
            utt_temp = save_result_train_temp[j+1].replace(save_result_train_temp[j],'')
            save_result_train.append(['utterance generation: '+tempt,utt_temp.split('[SEP]')[0]])
        
        

    ## test set
    for j in range(len(test_x)):
        tempt = ''
        if len(test_x[j]) == 0:
            save_result_test.append(['satisfaction score: '+ tempt,test_sat[j]+1])
            save_result_test.append(['action prediction: '+ tempt,test_act[j]])
            #save_result_test.append(['utterance generation: '+ tempt,''])
            continue
        for k in test_x[j]:
            tempt += tokenizer.decode(k)+' '

        save_result_test.append(['satisfaction score: '+ tempt,test_sat[j]+1])
        ## Action Prediction Data
        save_result_test.append(['action prediction: '+ tempt,test_act[j]])
        ## Utterance Generation Data
        
        if j+1 == len(test_x):
            continue
        if len(train_x[j+1]) == 0:
            continue
        else:
            utt_temp = save_result_test_temp[j+1].replace(save_result_test_temp[j],'')
            save_result_test.append(['utterance generation: '+ tempt,utt_temp.split('[SEP]')[0]])
        
    
    
    ## -- upsample the dataset, if needed
    ## -- Attention: the test set should not be augmented
    aug_result = augment_text(save_result_train)
    save_result_train = save_result_train + aug_result


    ## save the final dataset. The path should be edited
    with open('sat-act-utt-aug/MWOZ_train_'+str(fold)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['input_text','target_text'])
        writer.writerows(save_result_train)
    print('Done')
    
    with open('sat-act-utt-aug/MWOZ_test_'+str(fold)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['input_text','target_text'])
        writer.writerows(save_result_test)
    print('Done')



data_processing(fold=0, data_name='MWOZ')
