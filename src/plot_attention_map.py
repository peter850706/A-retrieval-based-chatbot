import argparse
import pdb
import sys
import pickle
import importlib
import torch
import torch.nn as nn
import nltk
import json
import os
import logging
import numpy
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    logging.info('Loading validation data...')
    with open('../data/valid.json', 'rb') as f:
        data = json.load(f)
    logging.info('Loading preproprecessed word embedding...')
    with open('../data/embedding.pkl', 'rb') as f:
        embedding = pickle.load(f)
    
    logging.info('Processing the context and the correct option...')
    data = data[16]
    nltk.download('punkt')
    def tokenize(sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens
    def sentence_to_indices(sentence, embedding):
        words = tokenize(sentence)
        indices = [embedding.to_index(word.lower()) for word in words]
        return words, indices

    context_tokens, context_indices = [], []
    options_tokens, options_indices = [], []
    for message in data['messages-so-far']:
        tokens, indices = sentence_to_indices(message['utterance'], embedding)
        context_tokens.append(tokens)
        context_indices.append(indices)    
    for option in data['options-for-correct-answers']:
        tokens, indices = sentence_to_indices(option['utterance'], embedding)
        options_tokens.append(tokens)
        options_indices.append(indices)
        
    data = {}
    speaker_tokens_indices = [embedding.to_index('</s1>'), embedding.to_index('</s2>')]
    context = []
    for i, utterance in enumerate(context_indices):
        context.extend([speaker_tokens_indices[i % 2]])
        context.extend(utterance)
    options = options_indices
    data['context'] = context
    data['options'] = options
    
    xticklabels, yticklabels = [], []
    speaker_tokens = ['</s1>', '</s2>']
    for i, context_token in enumerate(context_tokens):
        xticklabels.append(speaker_tokens[i % 2])
        xticklabels.extend(context_token)
    for option_token in options_tokens:    
        yticklabels.extend(option_token)
    
    
    if args.device is not None:
        device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        logging.warning('Not using GPU.')
        
    def get_instance(module, config, *args, **kwargs):
        return getattr(module, config['type'])(*args, **config['args'], **kwargs)

    path = args.model_dir
    with open(os.path.join(path, 'config.json'), 'rb') as f:
        config = json.load(f)
    arch = config['model_parameters']['arch']
    module_arch = importlib.import_module("modules." + arch['type'].lower())
    model = get_instance(module_arch, arch, dim_embedding=embedding.vectors.size(1))
    model_embedding = nn.Embedding(embedding.vectors.size(0),
                                   embedding.vectors.size(1))
    model_embedding.weight = nn.Parameter(embedding.vectors)

    checkpoint = torch.load(os.path.join(args.model_dir, 'model.best.pkl'))
    model.load_state_dict(checkpoint['model'])
    model_embedding.load_state_dict(checkpoint['embedding'])
    model = model.to(device)
    model_embedding = model_embedding.to(device)

    context_lens = torch.tensor([len(data['context'])]) 
    options_lens = torch.tensor([[len(option) for option in data['options']]])
    context = model_embedding(torch.tensor([data['context']]).to(device))
    options = model_embedding(torch.tensor([data['options']]).to(device))
    
    logging.info('Generating the attention map...')
    logits = model.forward(context,
                           context_lens.to(device),
                           options,
                           options_lens.to(device))
    attention_map = model.attention.option_attention_weights.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(22, 4))
    ax = sns.heatmap(attention_map, vmin=0, vmax=1, cmap='YlGnBu', annot=True, fmt ='.3f', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title("attention map")
    plt.xlabel("context")
    plt.ylabel("option")    
    figure = ax.get_figure()
    
    logging.info('Saving the attention map to /src/attention_map.png...')
    figure.savefig("attention_map.png")

def _parse_args():
    parser = argparse.ArgumentParser(description="Script to plot attention map.")
    parser.add_argument('model_dir', type=str, help='Directory to the model checkpoints and log.')
    parser.add_argument('--device', default=None, help='Device used to train. Can be cpu or cuda:0, cuda:1, etc.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)