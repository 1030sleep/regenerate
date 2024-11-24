import random
import json
def addition_input(sequence,num_1,num_2,train=True):
    # 为了方便，ntp模型训练只需要用到input
    length=len(num_1)
    if train==False:
        return sequence+'>'*length
    num_1=int(num_1,2)
    num_2=int(num_2,2)
    result=bin(num_1+num_2)[2:]
    input=sequence+(length+1-len(result))*'0'+result
    return input


def addition_output(sequence,num_1,num_2):
    length=len(num_1)
    num_1=int(num_1,2)
    num_2=int(num_2,2)
    result=bin(num_1+num_2)[2:]
    result=(length+1-len(result))*'0'+result
    output='*'*(len(sequence)-1)+result
    return output


def generate_addition_sequence(len):
    num_1=''
    num_2=''
    for i in range(len):
        num_1+=str(random.randint(0,1))
        num_2+=str(random.randint(0,1))
    sequence=num_1+'+'+num_2
    sequence+='>'
    return sequence,num_1,num_2

file_path='data/train/addition.json'

data=[]
for i in range(1,21):
    for j in range(102400):
        sequence,num_1,num_2=generate_addition_sequence(i)
        input=addition_input(sequence,num_1,num_2)
        output=addition_output(sequence,num_1,num_2)
        dict={}
        dict['input']=input
        dict['output']=output
        dict['length']=2*i+1
        dict['step']=i+1
        data.append(dict)

with open(file_path,"w",encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print('ok')

file_path_test='data/test/addition.json'
data_test=[]
for i in range(1,50):
    for j in range(64):
        sequence,num_1,num_2=generate_addition_sequence(i)
        input=addition_input(sequence,num_1,num_2,train=False)
        output=addition_output(sequence,num_1,num_2)
        dict={}
        dict['input']=input
        dict['output']=output
        dict['length']=2*i+1
        dict['step']=i+1
        data_test.append(dict)

with open(file_path_test,"w",encoding='utf-8') as json_file:
    json.dump(data_test, json_file, ensure_ascii=False, indent=4)

print('ok')
