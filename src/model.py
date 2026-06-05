# src/model.py
import torch
import torch.nn as nn
from transformers import BertModel

class BertBiGRU(nn.Module):
    def __init__(self, bert_model=None, hidden_size=128, num_classes=2):
        super(BertBiGRU, self).__init__()

        # Use a pretrained BERT encoder if provided
        self.bert = bert_model if bert_model else BertModel.from_pretrained(
            "bert-base-uncased",
            local_files_only=True,
            ignore_mismatched_sizes=True  # prevent UNEXPECTED keys warnings
        )

        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state

        gru_output, _ = self.gru(sequence_output)
        last_hidden = gru_output[:, -1, :]

        logits = self.fc(last_hidden)
        return logits