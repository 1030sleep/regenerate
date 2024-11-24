import random
import json
def mul_input(sequence,num_1,num_2,train=True):
    # 为了方便，ntp模型训练只需要用到input
    length=len(num_1)*2
    if train==False:
        return sequence+'>'*(length-1)
    num_1=int(num_1,2)
    num_2=int(num_2,2)
    result=bin(num_1*num_2)[2:]
    input=sequence+(length-len(result))*'0'+result
    return input


def mul_output(sequence,num_1,num_2):
    length=len(num_1)*2
    num_1=int(num_1,2)
    num_2=int(num_2,2)
    result=bin(num_1*num_2)[2:]
    result=(length-len(result))*'0'+result
    output='*'*(len(sequence)-1)+result
    return output


def generate_mul_sequence(len):
    num_1=''
    num_2=''
    for i in range(len):
        num_1+=str(random.randint(0,1))
        num_2+=str(random.randint(0,1))
    sequence=num_1+'*'+num_2
    sequence+='>'
    return sequence,num_1,num_2

file_path='data/train/mul.json'

data=[]
for i in range(1,11):
    for j in range(32000):
        sequence,num_1,num_2=generate_mul_sequence(i)
        input=mul_input(sequence,num_1,num_2)
        output=mul_output(sequence,num_1,num_2)
        dict={}
        dict['input']=input
        dict['output']=output
        dict['length']=2*i+1
        dict['step']=i*i
        data.append(dict)

with open(file_path,"w",encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print('ok')

file_path_test='data/test/mul.json'
data_test=[]
for i in range(1,50):
    for j in range(64):
        sequence,num_1,num_2=generate_mul_sequence(i)
        input=mul_input(sequence,num_1,num_2,train=False)
        output=mul_output(sequence,num_1,num_2)
        dict={}
        dict['input']=input
        dict['output']=output
        dict['length']=2*i+1
        dict['step']=i*i
        data_test.append(dict)

with open(file_path_test,"w",encoding='utf-8') as json_file:
    json.dump(data_test, json_file, ensure_ascii=False, indent=4)

print('ok')
