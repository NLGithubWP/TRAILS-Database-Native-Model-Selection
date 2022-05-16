import json
import logging

from sampler.random_sample import RandomSampler
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import argparse
import torch
from torch.autograd import Variable

from search_algorithm.nasi import NASIEvaluator
from search_space.NASBench101.apis import get_nasbench101_api
from search_space.NASBench101.query import Query101

logging.basicConfig(filename="score_log_101",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('Log')

def Test(net, testloader, criterion, num_tests, predict_net=None):
    net.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).cpu().sum().item()

        print('Testing: Loss=%.3f, Acc=%.3f(%d/%d)' %
              (test_loss/len(testloader), correct/num_tests, correct, num_tests))


def PrepareDataset(batch_size):
    print('--- Preparing CIFAR10 Data ---')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    print('--- CIFAR10 Data Prepared ---')

    return trainloader, len(trainset), testloader, len(testset)

def Train(net, trainloader, testloader, criterion, optimizer, scheduler, num_trains, num_tests, args):
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    for epoch in range(num_epochs):
        net.train()

        scheduler.step()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            # inputs, targets = Variable(inputs, Variable(targets.cuda())

            # forward
            outputs = net(inputs)

            # back-propagation
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).cpu().sum().item()

            print('Epoch=%d Batch=%d | Loss=%.3f, Acc=%.3f(%d/%d)' %
                  (epoch, batch_idx+1, train_loss/(batch_idx+1), correct/total, correct, total))

        # testing
        Test(net, testloader, criterion, num_tests)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NASBench')
    parser.add_argument('--module_vertices', default=7, type=int, help='#vertices in graph')
    parser.add_argument('--max_edges', default=9, type=int, help='max edges in graph')
    parser.add_argument('--available_ops', default=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                        type=list, help='available operations performed on vertex')
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
    parser.add_argument('--epochs', default=100, type=int, help='#epochs of training')
    parser.add_argument('--learning_rate', default=0.025, type=float, help='base learning rate')
    parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')
    parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
    parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
    parser.add_argument('--num_labels', default=10, type=int, help='#classes')

    parser.add_argument("--data", type=str, default="/Users/kevin/project_python/Fast-AutoNAS/data",
                        help="location of the data corpus")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--steps", type=int, default=200, help="steps to optimize architecture")
    parser.add_argument("--report_freq", type=float, default=10, help="report frequency")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
    parser.add_argument("--layers", type=int, default=9, help="total number of layers")
    parser.add_argument("--search_space", choices=["normal", "reduce"], default="normal")
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--gumbel", action="store_true", default=False, help="use gumbel")
    parser.add_argument("--adaptive", action="store_true", default=False, help="adaptive gap")
    parser.add_argument("--out_weight", action="store_true", default=False, help="search for output nodes")
    parser.add_argument("--init_alphas", type=float, default=0, help="initial weights of braches")
    parser.add_argument("--reg_weight", type=float, default=2, help="operation regularizer")
    parser.add_argument("--gap", type=float, default=100, help="gap for regularizer")
    parser.add_argument("--sparsity", type=float, default=0, help="sparsity for operations")
    parser.add_argument("--rand_label", action="store_true", default=False, help="use rand_labeld search")
    parser.add_argument("--rand_data", action="store_true", default=False, help="use rand_data data")
    parser.add_argument("--save", type=str, default="naspi", help="experiment name")
    parser.add_argument("--seed", type=int, default=5, help="exp seed")

    args = parser.parse_args()

    search_space = "nasbench101"

    query_apis = get_nasbench101_api('/Users/kevin/project_python/Fast-AutoNAS/data/nasbench_only108.pkl')
    # query_apis = get_nasbench101_api('/home/xingnaili/Fast-AutoNAS/data/nasbench_only108.pkl')

    visited = []

    plot_score = []
    train_accuracy = []
    validation_accuracy = []
    test_accuracy = []

    for i in range(200):

        try:
            sampler = RandomSampler()
            architecture, matrix, operations = sampler.sample_one_arch(search_space, args, query_apis)

            if str(matrix) in visited:
                continue
            visited.append(str(matrix))

            evaluator = NASIEvaluator()
            train_loader, num_trains, test_loader, num_tests = PrepareDataset(args.batch_size)
            score = evaluator.score(arch=architecture, pre_defined=args, train_data=train_loader)
            query101 = Query101()
            real_res, statics = query101.query_result( query_apis, matrix, operations)
            # logger.info("Evaluation matrix = "+ matrix)
            # logger.info("Evaluation operations = " + operations)
            logger.info("Evaluation score is " + str(score.data.numpy()))
            logger.info("Queried performance is " + str(statics))
            logger.info("Queried full-result is " + real_res)

            plot_score.append(score.data.numpy())
            train_accuracy.append(statics[1])
            validation_accuracy.append(statics[2])
            test_accuracy.append(statics[3])

        except Exception as e:
            logger.info("error" + str(e))

        logger.info("\n")

    logger.info("score list = ")

    logger.info(plot_score)

    logger.info(train_accuracy)
    logger.info(validation_accuracy)
    logger.info(test_accuracy)




