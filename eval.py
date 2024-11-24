from tokenizer import tokenizer
import config
from dataset import MyDataset
from torch.utils.data import DataLoader
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
import torch
from torch import nn
from tqdm import tqdm
from model import LOOP_TF

batch_size=64

mode='addition'
loop_model_path='output\Loop-TF-'+mode
vanilla_model_path='output\Vanilla-ntp-'+mode
dataset_path='data\\test\\'+mode+'.json'

loop_model=torch.load(loop_model_path).to(config.device)
vanilla_model=torch.load(vanilla_model_path).to(config.device)

testdataset=MyDataset(dataset_path)
testloader=DataLoader(dataset=testdataset,batch_size=batch_size,shuffle=False)

# query='01000001010101>>>>>>>>>>>>>>'
# input_ids=tokenizer.encode(query,return_tensors='pt').to(config.device)
# print(input_ids)
# loss,logits,_=model(input_ids,input_ids,14,14,14)
# print(logits[:,-1:,:])
# id=torch.argmax(logits,dim=-1).squeeze()
# print(tokenizer.convert_ids_to_tokens(id))

loop_acc_list=[]
vanilla_acc_list=[]
max_test_length=30
with torch.no_grad():
    T_max=50
    # length_a=1
    acc_num=0
    for i,data in enumerate(tqdm(testloader,leave=False)):
        if(i>max_test_length-1):
            break
        inputs,outputs,lengths,steps=data
        # print(inputs)
    
        inputs_ids=tokenizer.batch_encode_plus(inputs,return_tensors='pt')['input_ids'].to(config.device)
        outputs_ids=tokenizer.batch_encode_plus(outputs,return_tensors='pt')['input_ids'].to(config.device)
        # print(outputs_ids)
        length_q=lengths[0].item()

        if mode=='parity':
            length_a=1
        elif mode=='copy':
            length_a=length_q
        elif mode=='addition':
            length_a=(length_q+1)//2

        hidden_state=None
        loss_min=1e9
        max_confidence_output=None
    
        for j in range(T_max):
            _,output,hidden_state=loop_model(inputs_ids,inputs_ids,length_q,length_a,1,hidden_state=hidden_state)
            # print(output.shape)
            labels=torch.argmax(output,dim=-1)
            loss=nn.functional.cross_entropy(output.view(-1,config.vocab_size),labels.view(-1),reduction='mean')
            if loss<loss_min:
                loss_min=loss
                max_confidence_output=output
        
        ground_truth_output=outputs_ids[:,length_q+1:length_q+1+length_a].contiguous()
        # real_loss=nn.functional.cross_entropy(max_confidence_output.view(-1,config.vocab_size),ground_truth_output.view(-1),reduction='mean')
        # print(real_loss)
        max_confidence_output=torch.argmax(max_confidence_output,dim=-1)
        max_confidence_output=max_confidence_output.view(-1,length_a)
        
        

        acc=torch.eq(max_confidence_output,ground_truth_output)
        
        acc=torch.sum(acc==True,dim=-1)
        # print(acc)
        acc_num=torch.sum(torch.eq(acc,torch.tensor(length_a)))
        # print(acc_num)
        # acc_num=torch.sum(acc==True)
        accuracy=acc_num/batch_size
        loop_acc_list.append(accuracy.item())

with torch.no_grad():
    for i,data in enumerate(tqdm(testloader)):
        if i>max_test_length-1:
            break
        inputs,outputs,lengths,steps=data
        length_q=lengths[0].item()

        if mode=='parity':
            length_a=1
        elif mode=='copy':
            length_a=length_q
        elif mode=='addition':
            length_a=(length_q+1)//2

        inputs_ids=tokenizer.batch_encode_plus(inputs,return_tensors='pt')['input_ids'].to(config.device)
        outputs_ids=tokenizer.batch_encode_plus(outputs,return_tensors='pt')['input_ids'].to(config.device)
        inputs_ids=inputs_ids[:,:length_q+2]

        for j in range(length_a):
            # print(inputs_ids.shape)

            logits=vanilla_model(inputs_ids).logits
            output=torch.argmax(logits,dim=-1)
            output=output[:,-1]
            output=output.view(-1,1)

            # print(tokenizer.convert_ids_to_tokens(output.view(-1)))
            inputs_ids=torch.cat((inputs_ids,output),dim=-1)
        
        # print(inputs_ids.shape)

        predict_outputs_ids=inputs_ids[:,length_q+2:length_q+2+length_a].contiguous()
        ground_truth=outputs_ids[:,length_q+1:-1].contiguous()
        # print(tokenizer.convert_ids_to_tokens(ground_truth.view(-1)))
        ground_truth=ground_truth.view(-1,length_a)

        # acc=torch.eq(output,ground_truth)
        # acc_num=torch.sum(acc==True)
        # print(acc_num/batch_size)
        # print(tokenizer.convert_ids_to_tokens(predict_outputs_ids[0]))
        acc=torch.eq(predict_outputs_ids,ground_truth)
        acc=torch.sum(acc==True,dim=-1)
        acc_num=torch.sum(torch.eq(acc,torch.tensor(length_a)))
        accuracy=acc_num/batch_size
        vanilla_acc_list.append(accuracy.item())


import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

title=mode

fig, ax = plt.subplots()
x=np.arange(1,max_test_length+1)
ax.plot(x,loop_acc_list,label='LOOP-FAP')
ax.plot(x,vanilla_acc_list,label='NTP')
ax.set_xlabel('test length')
ax.set_ylabel('accuracy')
ax.set_title(mode)
ax.grid()
ax.legend()
plt.savefig('imgs\\'+mode+'.png')
plt.show()


        
