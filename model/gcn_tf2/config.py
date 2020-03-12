import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora', help='citeseer, cora, pubmed') #cora, citeseer,
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', default=0.01)
args.add_argument('--epochs', default=3000)
args.add_argument('--hidden1', default=32)
args.add_argument('--dropout', default=0.1)
args.add_argument('--weight_decay', default=5e-4)
args.add_argument('--early_stopping', default=30)
args.add_argument('--max_degree', default=8)

args.add_argument('--task_type', default='semi',help='with semi, we adpot the train_val_test split from Original GCN, with full, we use all node expcets val and test for training.')
args.add_argument('--whole_batch', default=False, help='use whole batch or mini batch')

args.add_argument('--train_batch_size', default=256)
args.add_argument('--test_batch_size', default=4096)
args.add_argument('--val_batch_size', default=4096)


args = args.parse_args()
print(args)
params = vars(args)
