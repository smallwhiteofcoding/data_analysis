
#读取生成摘要与参考摘要，计算ROUGE-L
from src.io import DecodeDataset, eval_coll_fn, SummRating
from src.predict import predict
test_loader = DataLoader(DecodeDataset("test", opt.data), collate_fn=eval_coll_fn(word2idx=word2idx, src_max_len=800),
                                  num_workers=opt.batch_workers,
                                  batch_size=16, pin_memory=True, shuffle=False)

overall_model.load_state_dict(torch.load('./models/train_model'))
overall_model.eval()
all_sample=predict(test_loader,overall_model,opt)
all_sample_test=[]
for i in range(len(all_sample)):
    all_sample_test.append([str(i),all_sample[i]])
with open(join("./result", 'submission.csv'),'w+') as csvfile:
    writer=csv.writer(csvfile,delimiter="\t")
    writer.writerows(all_sample_test)

    import pandas as pd
    from rouge import Rouge

    pd_output = pd.read_csv("./result/submission.csv", sep="\t", names=["index", "output"])
    pd_label = pd.read_csv("./datasets/test_label.csv", sep="\t", names=["index", "label"])

    output = pd_output.output
    label = pd_label.label

    rouge = Rouge()
    rouge_score = rouge.get_scores(output, label)

    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]
    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))