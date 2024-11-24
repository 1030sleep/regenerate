from tokenizer import tokenizer
import config
from dataset import MyDataset
from torch.utils.data import DataLoader
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
import torch
from torch import nn
from tqdm import tqdm
from model import LOOP_TF,gpt2config_vanilla_ntp

def count_params(model: nn.Module):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params

mode='mul'

model_depth=1
if mode=='copy':
    model_depth=2
elif mode=='addition':
    model_depth=3
elif mode=='mul':
    model_depth=4


epoch=1
batch_size=8
ACC_GRAD_STEP=8
lr=1e-4
mask_num=2 #the id of '>'

dataset_path='data/train/'+mode+'.json'
traindataset=MyDataset(dataset_path)
trainloader=DataLoader(dataset=traindataset,batch_size=batch_size,shuffle=False)

loop_tf_save_path='output/Loop-TF-'+mode
vanilla_ntp_save_path='output/Vanilla-ntp-'+mode


def train_looptf(looptf):

    optimizer_looptf=torch.optim.AdamW(looptf.parameters(),lr=lr,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler_looptf=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_looptf,T_max=epoch)
    # scheduler_looptf=torch.optim.lr_scheduler.LinearLR(optimizer_looptf,start_factor=1,end_factor=0.1,total_iters=len(trainloader)*(epoch-1))
    
    for j in range(epoch):
        total_loss=0
        acc_grad_step=0

        for i,data in enumerate(tqdm(trainloader,leave=False)):
            # if i>24000 and j==0:
            #     break
            # if i>0:
            #     break
            inputs,outputs,lengths,steps=data
        
            inputs_ids=tokenizer.batch_encode_plus(inputs,return_tensors='pt')['input_ids'].to(config.device)
            outputs_ids=tokenizer.batch_encode_plus(outputs,return_tensors='pt')['input_ids'].to(config.device)
            # print(tokenizer.convert_ids_to_tokens(inputs_ids[0]))
            
            length_q=lengths[0].item()
            step=steps[0].item()
            
            if mode=='copy':
                length_a=length_q
            elif mode == 'parity':
                length_a=1
            elif mode=='addition':
                length_a=(length_q+1)//2
            elif mode=='mul':
                length_a=length_q-1
            
    
            # for FAP mask the inputs_ids
            inputs_ids[:,length_q+2:length_q+2+length_a]=torch.tensor(mask_num)
       
    
            loss,logits,_=looptf(inputs_ids,outputs_ids,length_q,length_a,step)
        
            
            loss.backward()
            acc_grad_step+=1
            if acc_grad_step%ACC_GRAD_STEP==0:
                optimizer_looptf.step()
                optimizer_looptf.zero_grad()
            
            scheduler_looptf.step()
            

            total_loss+=loss.item()
            if (i+1)%1000==0:
                print(total_loss/1000)
                # print(torch.argmax(logits,dim=-1))
                total_loss=0
        
                torch.save(looptf,loop_tf_save_path)
        
        
        



def train_vanilla_ntp(vanilla_ntp):
    optimizer_vanilla=torch.optim.AdamW(vanilla_ntp.parameters(),lr=1e-4,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler_vanilla=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_vanilla,T_max=epoch)
    # scheduler_vanilla=torch.optim.lr_scheduler.LinearLR(optimizer_vanilla,start_factor=1,end_factor=0.1,total_iters=len(trainloader)*(epoch-1))
    
    for j in range(epoch):
        total_loss=0
        acc_grad_step=0
        for i,data in enumerate(tqdm(trainloader,leave=False)):
            inputs,outputs,lengths,steps=data
            inputs_ids=tokenizer.batch_encode_plus(inputs,return_tensors='pt')['input_ids'].to(config.device)
            outputs_ids=tokenizer.batch_encode_plus(outputs,return_tensors='pt')['input_ids']
    
            loss=vanilla_ntp(input_ids=inputs_ids,labels=inputs_ids).loss
            optimizer_vanilla.zero_grad()
            loss.backward()
            acc_grad_step+=1
            if acc_grad_step%ACC_GRAD_STEP==0:
                optimizer_vanilla.step()
                optimizer_vanilla.zero_grad()
            total_loss+=loss.item()
            if (i+1)%1000==0:
                print(total_loss/1000)
                total_loss=0
                torch.save(vanilla_ntp,vanilla_ntp_save_path)
            scheduler_vanilla.step()

# looptf=LOOP_TF(depth=model_depth).to(config.device)
# looptf=torch.load(loop_tf_save_path)
# train_looptf(looptf=looptf)
gpt2config_vanilla_ntp.n_layer=20*model_depth
vanilla_ntp=GPT2LMHeadModel(gpt2config_vanilla_ntp).to(config.device)
train_vanilla_ntp(vanilla_ntp=vanilla_ntp)


