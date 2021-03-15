import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import logging
from nerreader import PAD, UNK, START_TAG, END_TAG, PAD_IND, END_IND, START_IND, UNK_IND


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size, device):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size), requires_grad=True)
        self.transition.data.zero_()
        self.transition.data[START_IND, :] = torch.tensor(-10000)
        self.transition.data[:, END_IND] = torch.tensor(-10000)

    def forward(self, feats):
        """
        Forward propagation.
        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size()[0]
        self.timesteps = feats.size()[1]

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(3).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = torch.cat([emission_scores[:, 0, :, :].unsqueeze(1),
                                emission_scores[:, 1:, :, :] + self.transition.unsqueeze(0).unsqueeze(0)], dim=1)
        return crf_scores


class CRFLoss(nn.Module):

    def __init__(self, num_labels, start_tag="[CLS]", end_tag="[SEP]", device='cpu'):
        super(CRFLoss, self).__init__()
        self.device = device
        self.tagset_size = num_labels
        print("end index : {}".format(END_IND))
        print("start index : {}".format(START_IND))
        print("Tag set size : {}".format(self.tagset_size))
        self.START_TAG = start_tag
        self.END_TAG = end_tag

    def _log_sum_exp(self, tensor, dim):
        """
        Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.
        :param tensor: tensor
        :param dim: dimension to calculate log-sum-exp of
        :return: log-sum-exp
        """
        m, _ = torch.max(tensor, dim)
        m_expanded = m.unsqueeze(dim).expand_as(tensor)
        return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))

    ## gets crf scores shape (BATCH_SIZE, WORD_SIZES, TAG_SET, TAG_SET)
    def forward(self, scores, targets, lengths):
        """
            _log_sum_exp(all scores ) - gold_score

            lengths are used to ignore the loss at the paddings!!
            assumes that the sentences are sorted by length
        """

        ## this calculation assumes that the first target, i.e target[0]
        ## no it does no!
        # logging.info("Targets")
        # logging.info(targets[-1]//(scores.size()[2]))
        lengths = lengths
        targets = targets[:, 1:]
        scores = scores[:, 1:]
        lengths = torch.tensor([l - 1 for l in lengths])
        targets = targets.unsqueeze(2)

        batch_size = scores.size()[0]
        # scores_ =  scores.view(scores.size()[0],scores.size()[1],-1)
        score_before_sum = torch.gather(scores.reshape(scores.size()[0], scores.size()[1], -1), 2, targets).squeeze(2)
        score_before_sums = pack_padded_sequence(score_before_sum, lengths, batch_first=True)
        # print(score_before_sum[0])
        gold_score = score_before_sums[0].sum()

        ## forward score : initialize from start tag
        forward_scores = torch.zeros(batch_size, self.tagset_size).to(self.device)

        # forward_scores[:batch_size] = self._log_sum_exp(scores[:, 0, :, :], dim=2)
        ## burada  hangisi  dogru emin   degilim index1-> index2 or  opposite?
        ## i think  opposite  is correct
        forward_scores[:batch_size] = scores[:, 0, :, START_IND]
        ## forward score unsqueeze 2ydi 1 yaptim cunku ilk index next tag olarak
        ## kurguluyorum
        logging.info("Scores shape: {}".format(scores.shape))
        logging.info("Scores size: {}".format(scores.size()))
        logging.info("Lenghts: {}".format(lengths))

        for i in range(1, scores.size()[1]):
            batch_size_t = sum([1 if lengths[x] > i else 0 for x in range(lengths.size()[0])])
            logging.info("batch_size_t: {}".format(batch_size_t))
            logging.info("scores: {}".format(scores.shape))
            logging.info("forward_scores: {}".format(len(forward_scores)))
            forward_scores[:batch_size_t] = \
                self._log_sum_exp(scores[:batch_size_t, i, :, :] \
                                  + forward_scores[:batch_size_t].unsqueeze(1), dim=2)
        all_scores = forward_scores[:, END_IND].sum()
        loss = all_scores - gold_score
        # loss = loss/batch_size
        return loss
