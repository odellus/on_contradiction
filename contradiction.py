import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datasets
import pandas as pd
import numpy as np

from transformers import LongformerTokenizerFast, LongformerModel
from tqdm import tqdm

class ContradictionClassifier(nn.Module):
    """I needed to know what LongformerForSequenceClassification was doing
    """
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.fc1 = nn.Linear(1024, 3)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        global_attention_mask,
    ):
        x = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        ).pooler_output
        x = self.fc1(x)
        x = self.softmax(x)
        return x#.argmax(dim=-1)


def load_longformer():
    fpath = './models/longformer-large-4096'
    tokenizer = LongformerTokenizerFast.from_pretrained(fpath)
    model = LongformerModel.from_pretrained(fpath)
    return tokenizer, model

def init_contradiction_classifier(model):
    return ContradictionClassifier(model)

def test_classifier(input_str, tokenizer, classifier):
    inputs = tokenizer.batch_encode_plus(
        input_str, 
        return_tensors='pt',
        padding=True,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1
    return classifier(
        input_ids.to('cuda'), 
        attention_mask.to('cuda'), 
        global_attention_mask.to('cuda'),
    )


def construct_dataset(tokenizer, max_length):
    df_train = pd.read_csv('train.csv')
    # Worrying about test later
    # df_test = pd.read_csv('test.csv')
    df_train = get_english(df_train)
    ds = get_dataset(df_train)
    return make_numeric(ds, tokenizer, max_length)

def get_english(df):
    # We'll focus on contradictions in one language for now
    df = df[df['lang_abv'] == 'en']
    # Drop unnecessary columns
    df = df.drop(columns=['id','language', 'lang_abv'])
    return df

def get_dataset(df):
    ds = datasets.Dataset.from_pandas(df) 
    combined = [' '.join([x['premise'], x['hypothesis']]) for x in ds]
    ds = ds.add_column('combined', combined)
    ds = ds.remove_columns(
        column_names=[
            '__index_level_0__',
            'premise',
            'hypothesis',
        ]
    )
    return ds

def make_numeric(ds, tokenizer, max_length):
    def process_data(batch):
        inputs = tokenizer(
            batch['combined'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        batch['input_ids'] = inputs.input_ids.numpy()
        batch['attention_mask'] = inputs.attention_mask.numpy()
        batch_size = len(batch['input_ids'])
        seq_len = len(batch['input_ids'][0])
        batch['global_attention_mask'] = np.zeros_like(batch['input_ids'])
        batch['global_attention_mask'][:, 0] = 1
        return batch
    ds_numeric = ds.map(
        process_data, 
        batched=True,
        batch_size=16,
        remove_columns=['combined'],
    )
    
    ds_numeric.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'global_attention_mask', 'label'],
    )
    
    return ds_numeric

def get_optimizer(model, lr):
    return optim.AdamW(model.parameters(), lr=lr)


def get_num_batch(dataset, batch_size):
    n_samples = len(dataset)
    n_batch = n_samples // batch_size
    if n_samples % batch_size == 0:
        return n_batch
    else:
        return n_batch + 1

def get_batch(dataset, batch_size, batch_idx):
    n_samples = len(dataset)
    n_batch = get_num_batch(dataset, batch_size)
    assert batch_idx < n_batch
    if batch_idx == n_batch - 1:
        # This isn't necessary when n_samples % batch_size == 0
        # But it doesn't break anything either so it's fine
        batch = dataset.select(
            range(
                batch_size * batch_idx, 
                n_samples,
            )
        )
    else:
        batch = dataset.select(
            range(
                batch_size * batch_idx, 
                batch_size * (batch_idx + 1),
            )
        )
    return batch

def train(model, device, dataset, optimizer, batch_size):
    model.train()
    model.to(device)
    n_samples = len(dataset)
    n_batch = get_num_batch(dataset, batch_size)
    res = []
    for batch_idx in tqdm(range(n_batch)):
        batch = get_batch(dataset, batch_size, batch_idx)
        optimizer.zero_grad()
        output = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['global_attention_mask'].to(device),
        )
        target = batch['label'].to(device)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f'On batch {batch_idx} out of {n_batch}')
        print(f'Latest loss: {loss.item()}')
    torch.save(model.state_dict(), 'contradiction_classifier.pt')


            
def main():
    '''
    '''
    # Define some training arguments
    device = 'cuda'
    batch_size = 4
    max_length=1024
    learning_rate = 1e-4
    # Load pretrained tokenizer and model
    tokenizer, model = load_longformer()
    # Get the training dataset
    dataset = construct_dataset(tokenizer, max_length)
    
    # Add classifier layer to base model
    classifier = ContradictionClassifier(model)
    optimizer = get_optimizer(classifier, learning_rate)
    train(classifier, device, dataset, optimizer, batch_size)

if __name__ == '__main__':
    main()
