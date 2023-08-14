import os
import csv

def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def results_to_file(args, test_acc, test_std):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("="*20)
        print("Creat Resulrts File !!!")
        #os.mkdir('./results/{}'.format(args.data))

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result_{}.csv".format(
                            args.dataset, args.seed)

    headerList = ["Method","drop-path",
                "::::::::",
                "test_acc", "val_acc"]

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

        line = "{}, {}, :::::::::, {:.4f}, {:.4f}\n".format(
            args.model_type, args.drop_path,
            test_acc, test_std
        )
        f.write(line)