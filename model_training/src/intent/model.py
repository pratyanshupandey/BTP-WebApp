import torch
import torch.nn as nn
from transformers import BertModel

class TorchGRUIntent(nn.Module):
    def __init__(self, hidden_size, vocab_size) -> None:
        super(TorchGRUIntent, self).__init__()
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        self.GRU = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(in_features=768, out_features=vocab_size)
        # self.query = nn.Parameter(data = torch.empty(1, 2 * hidden_size))
        # nn.init.uniform_(self.query, -1, 1)
        # self.mha = nn.MultiheadAttention(embed_dim=2 * hidden_size ,num_heads=2, batch_first=True)
    
    def forward(self, ids, mask):

        # with torch.no_grad():
        x = self.bert_model(input_ids=ids, attention_mask=mask, output_hidden_states=True).last_hidden_state
            # x has shape N , L , 768

            # x = torch.transpose(x, 0 , 1)
            # # x has shape L, N , 768

        # out, h = self.GRU(x)
        out=x
        # outhas size L(seq_len),N(batch),D∗H_out(hidden)
        # h has shape D*num_layers, N, Hout(hidden size)​
        # h = torch.transpose(h, 0, 1)
        # # h has shape N, D*num_layers, Hout(hidden size)​
        # h = h.reshape(h.size()[0], -1)
        # outputs = self.classifier(torch.mean(out, dim=0))
        # batched_query = torch.stack([self.query] * out.size()[0]) 
        # print(batched_query.size())
        # print(out.size())
        # print((mask==False).size())
        # attn_out, _ = self.mha(query=batched_query, key=out, value=out, key_padding_mask=(mask==False))
        # outputs = self.classifier(attn_out.squeeze())
        outputs_tgt = self.classifier(torch.mean(out, dim=1))
        s = torch.unsqueeze(torch.mean(out, dim=1), 1)


        # outputs has shape N * vocab
        return outputs_tgt, torch.cat([s, s], dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
