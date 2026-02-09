from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments

curr_dir = Path(os.getcwd()).parent
with open(f"{curr_dir}/Data/train_processed", "rb") as f:
    dftr = pickle.load(f)

train_df, eval_df = train_test_split(
    dftr, test_size=0.2, stratify=dftr["label"], random_state=42
)
# Load model directly
from transformers import AutoTokenizer

from transformers import BertModel
import torch.nn as nn 
import matplotlib.pyplot as plt
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.dropout = nn.Dropout(0.1)  # regularization, default .5 is too high
        self.classifier = nn.Linear(768, 3)  # bert base 768, class of 3
        self.bert.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )  # bert output
        output = self.dropout(output.pooler_output)  # dropout
        logits = self.classifier(output)  # classifier

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.classifier.out_features), labels.view(-1)
            )  # loss calculation

        return loss, logits


model = (
    CustomBERTModel()
)  # import bert model and introduce random dropout and label classifier


# tokenize the data
def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding=True)
    return tokens


train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
# map the tokenize function to the dataset
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
eval_dataset = eval_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))

# set the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4.1,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=100,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    learning_rate=1.75e-5,  # learning rate
    logging_dir=f"{curr_dir}/Output/logs",
    logging_steps=10,  # logging steps
    metric_for_best_model="accuracy",  # Use accuracy for early stopping evaluation
)

# define the trainer
trainer = Trainer(
    model=model,  # the instantiated hf Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=eval_dataset,  # evaluation dataset
)

trainer.train()

training_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = list(range(len(training_loss)))

plt.figure()
plt.plot(steps, training_loss, label = "training_loss")



