import argparse
import csv
import os
import random
from importlib.metadata import version

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          LlamaTokenizer)

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + '.' + name1 if name != '' else name1))
    return res


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:

    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    # Load local dataset
    traindata = load_from_disk('./data/wikitext2_train')
    testdata = load_from_disk('./data/wikitext2_test')

    # Encode datasets
    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        # tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)


def get_llm(model, cache_dir='llm_weights'):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map='auto')
    print('printing gpu allocation for all the layers')
    print(model.hf_device_map)
    model.seqlen = 2048
    return model


class GradientComputation:

    def __init__(self, model, scale):
        self.model = model
        self.gradients_l1 = dict()
        self.gradients_l2 = dict()
        self.nsample = 0
        self.scale = scale
        self.device = torch.device('cpu')
        self.gradients_init()

    def gradients_init(self):
        if 'OPT' in self.model.model.__class__.__name__:
            layers = self.model.model.decoder.layers
        else:
            layers = self.model.model.layers

        for i in tqdm(
                range(len(layers)),
                desc=f'initializing the gradient list ....'):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f'{name}_layer_{i}'
                self.gradients_l1[indexed_name] = torch.zeros_like(
                    subset[name].weight,
                    dtype=torch.float16,
                    device=self.device)
                self.gradients_l2[indexed_name] = torch.zeros_like(
                    subset[name].weight,
                    dtype=torch.float32,
                    device=self.device)

    def update_gradient(self, model, nsample):
        assert nsample - self.nsample == 1, 'number of samples must be incremented by 1'
        if 'OPT' in model.model.__class__.__name__:
            layers = model.model.decoder.layers
        else:
            layers = model.model.layers
        for i in tqdm(
                range(len(layers)),
                desc=f'updating the gradient of sample no: {self.nsample}'):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f'{name}_layer_{i}'
                if subset[name].weight.grad is None:
                    print(f'Error: {name} has none gradient')
                if subset[name].weight.grad is not None:
                    assert subset[
                        name].weight.requires_grad == True, f'Required grad must be true ( {name}: {subset[name].weight.requires_grad})'
                    grad = subset[name].weight.grad.detach().clone().to(
                        dtype=torch.float32)  # Cast to float32
                    all_zero = (torch.abs(grad) == 0).all()
                    assert int(
                        all_zero
                    ) == 0, f'all the elements in the tensor are zero.: {all_zero}'
                    assert self.gradients_l1[
                        indexed_name].shape == grad.shape, 'shape mismatch'
                    self.gradients_l1[indexed_name] = self.gradients_l1[
                        indexed_name] + torch.abs(grad * self.scale).to(
                            device=self.device).to(dtype=torch.float16)
                    self.gradients_l2[indexed_name] = self.gradients_l2[
                        indexed_name] + torch.abs(
                            (grad * self.scale)**2).to(device=self.device)
        self.nsample = nsample


def get_activation(name, activations):
    """ Function to return a hook that stores the output of the layer in the provided dictionary. """

    def hook(model, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations[name] = output.detach()

    return hook


class ActivationComputation:

    def __init__(self, model):
        self.model = model
        self.activations = {}  # Store activations
        self.activations_l1 = dict()
        self.activations_l2 = dict()
        self.register_hooks()

    def register_hooks(self):
        if 'OPT' in self.model.model.__class__.__name__:
            layers = self.model.model.decoder.layers
        else:
            layers = self.model.model.layers 
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f'{name}_layer_{i}'
                layer.register_forward_hook(
                    get_activation(indexed_name, self.activations))

    def update_activation(self):
        for name, activation in self.activations.items():
            self.activations_l1[name] = torch.abs(activation).mean(
                dim=0, keepdim=True).type(torch.float16)
            self.activations_l2[name] = (activation**2).mean(
                dim=0, keepdim=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nsamples', type=int, default=2, help='no of samples used')
    parser.add_argument(
        '--scale', type=int, default=100, help='no of samples used')
    parser.add_argument(
        '--llama_version', type=int, default=2, help='llama version used')
    parser.add_argument('--model', type=str, help='model to used')
    parser.add_argument(
        '--task', type=str, default='gradient', help='task to be performed')
    parser.add_argument('--seed', type=int, default=0, help='seed used')
    args = parser.parse_args()
    print(
        f'Obtaining gradients for no of samples {args.nsamples}, scale {args.scale}'
    )

    model_args = args.model
    cache_dir_args = 'llm_weights'
    model = get_llm(model_args, cache_dir_args)
    if args.llama_version == 2:
        tokenizer = AutoTokenizer.from_pretrained(model_args, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_args, use_fast=False)

    if 'opt' in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    if 'model.embed_tokens' in model.hf_device_map:
        device = model.hf_device_map['model.embed_tokens']
    else:
        device = model.device
    
    
    print('loading calibdation data')
    nsamples = args.nsamples
    seed = args.seed
    dataloader, _ = get_loaders(
        'wikitext2',
        nsamples=nsamples,
        seed=seed,
        seqlen=2048,
        tokenizer=tokenizer)

    print('dataset loading complete')
    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    optimizer.zero_grad()

    if args.task == 'gradient':
        computer = GradientComputation(model, args.scale)
    elif args.task == 'activation':
        computer = ActivationComputation(model)
    else:
        raise ValueError(f'task {args.task} not supported')

    nsample = 0
    model.train()

    for input_ids, labels in tqdm(dataloader):
        nsample += 1
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        # print('Printing the loss:', loss)
        loss.backward()

        if args.task == 'gradient':
            computer.update_gradient(model, nsample)
        elif args.task == 'activation':
            computer.update_activation()

        optimizer.zero_grad()
    print('Done')

    model_name = os.path.basename(args.model)

    if args.task == 'gradient':
        gradients_l2 = computer.gradients_l2
        for name in gradients_l2:
            grad_sqrt = torch.sqrt(gradients_l2[name])
            gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)

        if 'opt' in args.model:
            if not os.path.exists('./gradients/opt'):
                os.makedirs(f'./gradients/opt')

            with open(f'./gradients/opt/gradients_aggregate_norm_l2_{model_name}.path', 'wb') as f:
                torch.save(computer.gradients_l2, f)
        else:
            if not os.path.exists(f'./gradients/llama{args.llama_version}'):
                os.makedirs(f'./gradients/llama{args.llama_version}')
            with open(
                    f'./gradients/llama{args.llama_version}/gradients_aggregrate_norm_l2_model_{model_name}_{args.nsamples}_{args.seed}.pth',
                    'wb') as f:
                torch.save(computer.gradients_l2, f)
            with open(
                    f'./gradients/llama{args.llama_version}/gradients_aggregrate_norm_l1_model_{model_name}_{args.nsamples}_{args.seed}.pth',
                    'wb') as f:
                torch.save(computer.gradients_l1, f)

    elif args.task == 'activation':
        activations_l1 = computer.activations_l1
        activations_l2 = computer.activations_l2
        for name in activations_l1:
            activations_l1[name] = activations_l1[name].to(dtype=torch.float16)
            activations_l2[name] = activations_l2[name].to(dtype=torch.float16)
        if not os.path.exists(f'./activations/llama{args.llama_version}'):
            os.makedirs(f'./activations/llama{args.llama_version}')
        with open(
                f'./activations/llama{args.llama_version}/activations_aggregrate_norm_l2_model_{model_name}.pth',
                'wb') as f:
            torch.save(activations_l2, f)
        with open(
                f'./activations/llama{args.llama_version}/activations_aggregrate_norm_l1_model_{model_name}.pth',
                'wb') as f:
            torch.save(activations_l1, f)
