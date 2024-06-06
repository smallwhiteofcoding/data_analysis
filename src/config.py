#读取参数，由于参数较多，故将config.py文件放置在./src/下面。config.py中有训练与测试的相关参数（文件中有说明），可自行调整
parser = argparse.ArgumentParser(description='train_ml.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.train_ml_opts(parser)
    opt = parser.parse_args(args=[])
    opt = train_process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

