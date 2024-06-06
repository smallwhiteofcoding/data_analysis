#读取测试集数据，加载保存好的模型，进行预测，结果保存在./result/下面。由于predict.py涉及代码多，故放在./src下面
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