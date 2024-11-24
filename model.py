import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy
from typing import Optional
import config
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
def count_params(model: nn.Module):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params

gpt2config_looptf=GPT2Config(
    vocab_size=config.vocab_size,
    n_positions=256,
    n_embd=256,
    n_layer=2,
    n_head=8,
    bos_token_id=6,#cls
    eos_token_id=4,#sep
)
Loop_TF_Block=GPT2LMHeadModel(gpt2config_looptf)

class LOOP_TF(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.embedding=nn.Embedding(config.vocab_size,256)
        self.gpt2config=GPT2Config(
                                    vocab_size=config.vocab_size,
                                    n_positions=256,
                                    n_embd=256,
                                    n_layer=depth,
                                    n_head=8,
                                    bos_token_id=6,#cls
                                    eos_token_id=4,#sep
                                )
        self.Loop_Block=GPT2LMHeadModel(self.gpt2config)
    
    def forward(self,input_ids,output_ids,length_q,length_a,step,hidden_state=None,):
        input_emb=self.embedding(input_ids)
        position_ids=torch.zeros_like(input_ids)
        # print(input_emb.shape)
        if hidden_state == None:
            hidden_state=torch.zeros_like(input_emb)
        for i in range(step):
            hidden_state_with_injection=hidden_state+input_emb
            # print(hidden_state_with_injection.shape)
            output=self.Loop_Block(inputs_embeds=hidden_state_with_injection,output_hidden_states=True,position_ids=position_ids)
            logits=output.logits
            hidden_state=output.hidden_states[-1]
        
        # fap logits只保留预测结果
        # print(length_q)
        # print(logits.shape)
        logits=logits[:,length_q+1:length_q+length_a+1,:].contiguous()
        labels=output_ids[:,length_q+1:length_q+length_a+1].contiguous()
        # print(labels)
        # print(logits.shape,labels.shape)
        loss=F.cross_entropy(logits.view(-1,config.vocab_size),labels.view(-1),reduction='mean')
        # print(logits.shape)
        return loss,logits,hidden_state


gpt2config_vanilla_ntp=GPT2Config(
    vocab_size=config.vocab_size,
    n_positions=256,
    n_embd=256,
    n_layer=40,
    n_head=8,
    bos_token_id=6,#cls
    eos_token_id=4,#sep
)
Vanilla_NTP=GPT2LMHeadModel(gpt2config_vanilla_ntp)

if __name__=='__main__':
    output=Vanilla_NTP(torch.tensor([[1,3,5]]),output_hidden_states=True)
    print(output.hidden_states[-1])


        

