import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_roberta import RobertaModel, RobertaClassificationHead


class RobertaInceptionForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig

    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.tfidf = None
        self.linear_hidden_size = 2000
        self.max_length = config.max_length
        self.classifier_0 = nn.Linear(self.linear_hidden_size, self.num_labels)
        self.classifier_1 = nn.Linear(self.linear_hidden_size, self.num_labels)

        from .InceptionBlock import InceptionCNN, parse_opt
        opt = parse_opt()
        opt.label_size = self.num_labels
        opt.embedding_dim = self.hidden_size
        opt.max_seq_len = self.max_length
        opt.linear_hidden_size = self.linear_hidden_size
        self.in_cnn = InceptionCNN(opt)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            tfidf=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        self.tfidf = tfidf

        last_layer_output = outputs[2][-1]

        batch_layer_out_tfidf = torch.zeros_like(last_layer_output)
        i = 0
        for l_out, tf in zip(last_layer_output, tfidf):
            l_out_tf = torch.mm(tf, l_out)  # (1,hidden_size)
            tf = tf.view(-1, 1)  # (seq_length,1)
            weighted_out = torch.mm(tf, l_out_tf)  # (seq_length, hidden_size)

            batch_layer_out_tfidf[i] = weighted_out
            i += 1

        batch_layer_out_tfidf = torch.as_tensor(batch_layer_out_tfidf)

        out0 = last_layer_output
        out1 = batch_layer_out_tfidf

        out0 = self.in_cnn(out0)
        out1 = self.in_cnn(out1)

        logits0 = self.classifier_0(out0)
        logits1 = self.classifier_1(out1)

        logits = torch.mean(torch.stack([logits0, logits1]), 0)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = [loss, logits]
        else:
            outputs = [logits, ]
        return outputs  # (loss), logits, (hidden_states), (attentions)
