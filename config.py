from tokenizer import tokenizer
import torch

epoch=5
warmup_steps = 10
batch_size = 64
gradient_accumulation_steps = 4
lr = 1e-4
weight_decay = 0.1
max_seq_length = 256
adam_beta1 = 0.9
adam_beta2 = 0.99
lr_scheduler_type = "cosine"


"""The config file of the model."""
n_layer = 4
n_head = 4
n_vocab = 10
embedding_dim = 128
hidden_dim = 512
p = 0.1
# n_positions = 256

"""Others"""
device = "cuda" if torch.cuda.is_available() else "cpu" # 暂时单卡训练
vocab_size = 60