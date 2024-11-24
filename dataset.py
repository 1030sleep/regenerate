from torch.utils.data import Dataset
import torch
import json

class MyDataset(Dataset):
    def __init__(self,data_file_path):
        self.data_file_path=data_file_path
        self.inputs=[]
        self.outputs=[]
        self.lengths=[]
        self.steps=[]

        with open(self.data_file_path,'r',encoding='UTF-8') as f:
            data_load=json.load(f)
        for sample in data_load:
            self.inputs.append(sample['input'])
            self.outputs.append(sample['output'])
            self.lengths.append(sample['length'])
            self.steps.append(sample['step'])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index=index.tolist()
        
        batch_input=self.inputs[index]
        batch_output=self.outputs[index]
        batch_length=self.lengths[index]
        batch_step=self.steps[index]


        return batch_input,batch_output,batch_length,batch_step
    
traindataset=MyDataset('data/train/parity.json')
testdataset=MyDataset('data/test/parity.json')

if __name__=='__main__':
    from torch.utils.data import DataLoader
    trainloader=DataLoader(traindataset,batch_size=4)
    
    for i,data in enumerate(trainloader):
        if i>1:
            break
        inputs,outputs,lengths,steps=data
        
        inputs=list(inputs)
        print(inputs)