#定义RNNEncoderBasic和RNNDecoder



class RNNEncoder(nn.Module):
    """
    Base class for rnn encoder
    """

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        raise NotImplementedError


class RNNEncoderBasic(RNNEncoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoderBasic, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        # self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """

        src_embed = self.embedding(src)  # [batch, src_len, embed_size]

        # sort src_embed according to its length
        batch_size = src.size(0)
        assert len(src_lens) == batch_size
        sort_ind = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens_sorted = [src_lens[i] for i in sort_ind]
        src_embed = reorder_sequence(src_embed, sort_ind, batch_first=True)
        # src_embed_sorted_np = src_embed.detach().cpu().numpy()[:, 0, :]

        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens_sorted, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # restore the order of memory_bank
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        memory_bank = reorder_sequence(memory_bank, reorder_ind, batch_first=True)
        encoder_final_state = reorder_gru_states(encoder_final_state, reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)  # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        return (memory_bank.contiguous(), None), (encoder_last_layer_final_state, None)


def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first, [B, T, D] if batch first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]
    device = sequence_emb.device
    order = torch.LongTensor(order)
    order = order.to(device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_


def reorder_gru_states(gru_state, order):
    """
    gru_state: [num_layer * num_directions, batch, hidden_size]
    order: list of sequence length
    """
    assert len(order) == gru_state.size(1)
    order = torch.LongTensor(order).to(gru_state.device)
    sorted_state = gru_state.index_select(index=order, dim=1)
    return sorted_state


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, coverage_attn, copy_attn,
                 review_attn, pad_idx, attn_mode, dropout=0.0, rating_memory_pred=False):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = nn.Dropout(dropout)
        self.coverage_attn = coverage_attn
        self.copy_attn = copy_attn
        self.review_attn = review_attn
        self.pad_token = pad_idx
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.input_size = embed_size
        self.hr_enc = False

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=False, batch_first=False, dropout=dropout)
        self.attention_layer = Attention(
            decoder_size=hidden_size,
            memory_bank_size=memory_bank_size,
            coverage_attn=coverage_attn,
            attn_mode=attn_mode
        )

        merged_memory_bank_size = memory_bank_size

        self.rating_memory_pred = rating_memory_pred
        if self.rating_memory_pred:
            self.rating_attention_layer = Attention(
                decoder_size=hidden_size,
                memory_bank_size=embed_size,
                coverage_attn=coverage_attn,
                attn_mode=attn_mode)
            merged_memory_bank_size += embed_size

        if copy_attn:
            p_gen_input_size = embed_size + hidden_size + merged_memory_bank_size
            """
            if goal_vector_mode == 2:
                p_gen_input_size += goal_vector_size
            """
            self.p_gen_linear = nn.Linear(p_gen_input_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.vocab_dist_linear_1 = nn.Linear(hidden_size + merged_memory_bank_size, hidden_size)
        self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, h, memory_banks, src_masks, max_num_oovs, src_oov, coverage, decoder_memory_bank=None,
                rating_memory_bank=None, goal_vector=None):
        """
        :param y: [batch_size]
        :param h: [num_layers, batch_size, decoder_size]
        :param memory_banks: ([batch_size, max_src_seq_len, memory_bank_size], None)
        :param src_masks: ([batch_size, max_src_seq_len], None)
        :param max_num_oovs: int
        :param src_oov: [batch_size, max_src_seq_len]
        :param coverage: [batch_size, max_src_seq_len]
        :param decoder_memory_bank: [batch_size, t-1, decoder_size]
        :param rating_memory_bank: a FloatTensor, [batch, rating_v_size, emb_size]
        :param goal_vector: [1, batch_size, goal_vector_size]
        :return:
        """
        # use a consistent interface with HirEncRNNDecoder
        memory_bank = memory_banks[0]
        src_mask = src_masks[0]

        batch_size, max_src_seq_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        assert h.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]

        rnn_input = y_emb

        _, h_next = self.rnn(rnn_input, h)

        assert h_next.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        last_layer_h_next = h_next[-1, :, :]  # [batch, decoder_size]

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        word_context, word_attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask,
                                                                      coverage)
        # context: [batch_size, memory_bank_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        assert word_context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert word_attn_dist.size() == torch.Size([batch_size, max_src_seq_len])

        context = word_context
        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)  # [B, memory_bank_size + decoder_size]
        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))

        p_gen = None
        if self.copy_attn:
            """
            if self.goal_vector_mode == 2:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0), goal_vector.squeeze(0)), dim=1)  # [B, memory_bank_size + decoder_size + embed_size + goal_vector]
            else:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
            """
            p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
            p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * word_attn_dist

            if max_num_oovs > 0:
                # extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

            final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        return final_dist, h_next, word_context, word_attn_dist, p_gen, coverage


