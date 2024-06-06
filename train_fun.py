
#定义训练函数
import math
import torch
import os

EPS = 1e-8



def train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt):
    print('======================  Start Training  =========================')

    total_batch = 0
    early_stop_flag = False

    best_valid_joint_loss = float('inf')
    num_stop_dropping = 0

    previous_valid_loss = float('inf')
    overall_model.train()

    for epoch in range(0, opt.epochs):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            if early_stop_flag:
                break
            total_batch += 1

            # Training
            train_loss = train_one_batch(batch, overall_model, optimizer_ml, opt)
            if total_batch % opt.checkpoint_interval == 0:
                print('epoch:', epoch, '  ', 'total_batch', total_batch)
            if epoch >= 1:
                if total_batch % opt.checkpoint_interval == 0:
                    print('start valid')
                    valid_loss = evaluate_loss(valid_data_loader, overall_model, opt)
                    print('valid_loss', valid_loss)
                    overall_model.train()

                    if epoch >= opt.start_decay_and_early_stop_at:
                        if valid_loss < previous_valid_loss:  # update the best valid loss and save the model parameters
                            previous_valid_loss = valid_loss
                            print("Valid loss drops")

                            num_stop_dropping = 0
                            check_pt_model_path = os.path.join(opt.model_path, 'train_model')
                            torch.save(  # save model parameters
                                overall_model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            print('Saving checkpoint to %s' % check_pt_model_path)
                        else:
                            num_stop_dropping += 1
                            # decay the learning rate by a factor
                            if opt.learning_rate_decay < 1:
                                for i, param_group in enumerate(optimizer_ml.param_groups):
                                    old_lr = float(param_group['lr'])
                                    new_lr = old_lr * opt.learning_rate_decay
                                    if new_lr < opt.min_lr:
                                        new_lr = opt.min_lr
                                    if old_lr - new_lr > EPS:
                                        param_group['lr'] = new_lr

                        if not opt.disable_early_stop:
                            print("num_stop_dropping", num_stop_dropping)
                            if num_stop_dropping >= opt.early_stop_tolerance:
                                print('Have increased for %d check points, early stop training' % num_stop_dropping)
                                early_stop_flag = True
                                break


def train_one_batch(batch, overall_model, optimizer, opt):
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
    """
    trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[io.SEP_WORD]
                 if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eos>
    trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
    """
    batch_size = src.size(0)
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)

    src_sent_positions = src_sent_positions.to(opt.device)
    src_sent_mask = src_sent_mask.to(opt.device)

    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    optimizer.zero_grad()

    decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage = \
        overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums,
                      src_sent_mask)

    if decoder_dist is not None:
        if opt.copy_attention:  # Compute the loss using target with oov words
            generation_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                                   opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                   opt.lambda_coverage, opt.coverage_loss)
        else:  # Compute the loss using target without oov words
            generation_loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                                   opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                   opt.lambda_coverage, opt.coverage_loss)
    else:

        generation_loss = torch.Tensor([0.0]).to(opt.device)

    if math.isnan(generation_loss.item()):
        raise ValueError("Generation loss is NaN")

    # normalize generation loss
    total_trg_tokens = sum(trg_lens)
    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        generation_loss_normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        generation_loss_normalization = batch_size
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert generation_loss_normalization > 0, 'normalization should be a positive number'
    normalized_generation_loss = generation_loss.div(generation_loss_normalization)
    # back propagation on the joint loss
    normalized_generation_loss.backward()
    optimizer.step()
    return normalized_generation_loss.item()
