#建立词典，读取数据，初始化Seq2SeqModel。将模型放到GPU上去，打印模型，训练
# construct vocab
with open(join(opt.data, 'vocab_cnt.pkl'), 'rb') as f:
    wc = pkl.load(f)
word2idx, idx2word = io.make_vocab(wc, opt.v_size)
opt.word2idx = word2idx
opt.idx2word = idx2word
opt.vocab_size = len(word2idx)
# construct train_data_loader, valid_data_loader

train_data_loader, valid_data_loader = build_loader(opt.data, opt.batch_size, word2idx, opt.src_max_len, opt.trg_max_len, opt.batch_workers)

# construct model

overall_model = Seq2SeqModel(opt)
overall_model.to(opt.device)

# construct optimizer
optimizer_ml = torch.optim.Adam(params=filter(lambda p: p.requires_grad, overall_model.parameters()), lr=opt.learning_rate)

print(overall_model)
train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt)#如果只是想预测，可加载训练好的模型（只保存了模型参数），注释该行。

