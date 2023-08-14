import os
import csv

def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def results_to_file(args, test_acc, test_std, model_para):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("="*20)
        print("Creat Resulrts File !!!")
        #os.mkdir('./results/{}'.format(args.data))

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result_{}.csv".format(
                            args.dataset, args.seed)

    headerList = ["Method","final-epoch", "update-freq","final-density",
                "::::::::",
                "test_acc", "val_acc", "num_para"]

    #filename = "./results/{}/{}_{}_{}_{}_result.csv".format(sparse_way, args.model, args.data, args.final_density, args.final_density_adj)
    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if  header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                        fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {},  :::::::::, {:.4f}, {:.4f}, {}\n".format(
            args.model_type, args.final_prune_epoch, args.update_frequency,
            args.final_density,
            test_acc, test_std, model_para
        )
        f.write(line)