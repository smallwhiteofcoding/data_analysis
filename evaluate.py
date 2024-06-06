#定义运行训练验证集函数，数据放到GPU上

def build_loader(data_path, batch_size, word2idx, src_max_len, trg_max_len, num_workers):
    coll_fn_customized = io.summ_rating_flatten_coll_fn(word2idx=word2idx, src_max_len=src_max_len,
                                                        trg_max_len=trg_max_len)

    train_loader = DataLoader(SummRating('train', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=True)

    valid_loader = DataLoader(SummRating('val', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=False)
    return train_loader, valid_loader





9
def evaluate_loss(data_loader, overall_model, opt):
    overall_model.eval()
    generation_loss_sum = 0.0
    total_trg_tokens = 0
    total_num_iterations = 0

    batch_number = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            batch_number += 1
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src_mask = batch['src_mask']
            src_sent_positions = batch['src_sent_positions']
            src_sent_nums = batch['src_sent_nums']
            src_sent_mask = batch['src_sent_mask']
            src_oov = batch['src_oov_tensor']
            oov_lists = batch['oov_lists']
            src_str_list = batch['src_list_tokenized']
            trg_sent_2d_list = batch['tgt_sent_2d_list']
            trg = batch['tgt_tensor']
            trg_oov = batch['tgt_oov_tensor']
            trg_lens = batch['tgt_lens']
            trg_mask = batch['tgt_mask']

            indices = batch['original_indices']

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            total_num_iterations += 1
            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_sent_positions = src_sent_positions.to(opt.device)
            src_sent_mask = src_sent_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)
            decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage = overall_model(
                src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums,
                src_sent_mask)

            if decoder_dist is not None:
                if opt.copy_attention:  # Compute the loss using target with oov words
                    generation_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                                           opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                           opt.lambda_coverage, coverage_loss=False)
                else:  # Compute the loss using target without oov words
                    generation_loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                                           opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                           opt.lambda_coverage, coverage_loss=False)
            else:
                generation_loss = torch.Tensor([0.0]).to(opt.device)

            # normalize generation loss
            num_trg_tokens = sum(trg_lens)
            normalized_generation_loss = generation_loss.div(num_trg_tokens)

            generation_loss_sum += normalized_generation_loss.item()

    return normalized_generation_loss.item() / batch_number
