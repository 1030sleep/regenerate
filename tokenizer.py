from transformers import BertTokenizer
tokenizer =BertTokenizer(vocab_file='data/vocab.txt')
# tokenizer.add_tokens(str(0))
# tokenizer.add_tokens(str(1))
for i in range(10):
    tokenizer.add_tokens(str(i))
tokenizer.add_tokens('+')
for i in range(ord('a'),ord('z')+1):
    tokenizer.add_tokens(chr(i))
# bos_token = '[BOS]'
# eos_token = '[EOS]'
# tokenizer.add_special_tokens({
#     'bos_token': bos_token,
#     'eos_token': eos_token
# })

if __name__ == '__main__':
    text='01>'
    input_ids=tokenizer(text)['input_ids']
    print(input_ids)
    print(tokenizer.convert_ids_to_tokens(input_ids))
    print(tokenizer.cls_token_id)
    print(tokenizer.sep_token_id)
    print(tokenizer.get_added_vocab())