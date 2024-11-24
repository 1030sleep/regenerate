import random
import json
def parity_input(sequence,train=True):
    # 为了方便，ntp模型训练只需要用到input
    if train==False:
        return sequence
    input=sequence+sequence[-2]
    return input


def parity_output(sequence):
    output='*'*(len(sequence)-1)+sequence[-2]
    return output


def generate_parity_sequence(len):
    sequence=''
    for i in range(len):
        sequence+=str(random.randint(0,1))
    sequence+='>'
    return sequence

file_path='data/train/parity.json'

data=[]
for i in range(1,21):
    for j in range(64000):
        sequence=generate_parity_sequence(i)
        input=parity_input(sequence)
        output=parity_output(sequence)
        dict={}
        dict['input']=input
        dict['output']=output
        dict['length']=i
        dict['step']=i
        data.append(dict)

with open(file_path,"w",encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print('ok')

file_path_test='data/test/parity.json'
data_test=[]
for i in range(1,101):
    for j in range(64):
        sequence=generate_parity_sequence(i)
        input=parity_input(sequence,train=False)
        output=parity_output(sequence)
        dict={}
        dict['input']=input
        dict['output']=output
        dict['length']=i
        dict['step']=i
        data_test.append(dict)

with open(file_path_test,"w",encoding='utf-8') as json_file:
    json.dump(data_test, json_file, ensure_ascii=False, indent=4)

print('ok')


