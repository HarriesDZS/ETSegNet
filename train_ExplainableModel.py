# -*-coding:utf-8-*-
"""
训练同步分割和分类的网络
"""

from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import torch
from tqdm import tqdm
import numpy as np


from STSDataLoaderWithSUV import STSDataLoader #replace with your dataloader
from ExplainModel import ExplainMode
from Loss_functions import AutomaticWeightedLoss
from train_loss import RecallTverskyLoss,StandardSegLoss
from transform import MedicalTransformCompose

device_id = 0


def train(data_loader, net, criterion, optimizer):
    """
    模型训练的具体方法
    :param data_loader:
    :param net:
    :param scheduler:
    :param early_stopping:
    :param criterion
    :return:
    """
    tbar = tqdm(data_loader, ascii=True, desc='train', dynamic_ncols=True)
    for batch_idx,(case_id,  pet_data, ct_data, suv_data,  label) in enumerate(tbar):
        pet_data = pet_data.cuda(device=device_id)
        ct_data = ct_data.cuda(device=device_id)
        label = label.type(torch.LongTensor)
        label = label.cuda(device=device_id)
        suv_data = suv_data.cuda(device=device_id)
        pet_output, ct_output, fusion_seg, atten_map = net(ct_data, pet_data, suv_data)

        pet_loss = criterion[0](pet_output, label)
        ct_loss = criterion[1](ct_output, label)
        fusion_loss = criterion[2](fusion_seg, label)

        loss = criterion[3](pet_loss, ct_loss, fusion_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        tbar.set_postfix({"loss":loss.item(), "pet_loss":pet_loss.item(), "ct_loss":ct_loss.item(),
                          "fusion_loss":fusion_loss.item()})
        assert loss.item() > 0 , "loss is None"
        tbar.update(1)


def evaluate(data_loader, net, criterion, type):
    """
    验证数据集的方法
    :param data_loader:
    :param net:
    :param criterion:
    :param type:
    :return:
    """

    tbar = tqdm(data_loader, ascii=True, desc="[EVAL]{}".format(type), dynamic_ncols=True)
    anchor_case_id = -1
    predicts = []
    labels = []
    loss_list = []
    dice_list = []
    case_list = []
    recall_list = []
    precision_list = []


    for batch_idx, (case_id,  pet_data, ct_data, suv_data,  label) in enumerate(tbar):
        pet_data = pet_data.cuda(device=device_id)
        ct_data = ct_data.cuda(device=device_id)
        label = label.type(torch.LongTensor)
        label = label.cuda(device=device_id)
        suv_data = suv_data.cuda(device=device_id)
        pet_output, ct_output, fusion_seg, atten_map = net(ct_data, pet_data, suv_data)

        loss = criterion[2](fusion_seg, label)


        loss_list.append(loss.item())
        seg_predict = fusion_seg.cpu().detach().numpy().squeeze(1)
        label = label.cpu().detach().numpy()
        #predict[predict == 1] = 0
        #label[label == 1] = 0
        seg_predict[seg_predict >= 0.5] = 1
        seg_predict[seg_predict < 0.5] = 0
        label[label >= 1] = 1


        for i in range(len(case_id)):
            case_id_item = case_id[i]
            seg_predict_item = seg_predict[i]
            label_item = label[i]


            if anchor_case_id != -1 and anchor_case_id != case_id_item:
                predict_array = np.stack(predicts, axis=0)
                label_array = np.stack(labels, axis=0)
                dice = 2 * (predict_array * label_array).sum() / (predict_array.sum() + label_array.sum())
                recall = (predict_array[label_array == 1] == 1).sum() / (label_array == 1).sum()
                precision = (predict_array[label_array == 1] == 1).sum() / ((predict_array == 1).sum()+0.001)
                dice_list.append(dice)
                recall_list.append(recall)
                precision_list.append(precision)
                case_list.append(anchor_case_id)
                predicts.clear()
                labels.clear()
                # print(anchor_case_id, case_id_item, dice)
            anchor_case_id = case_id_item
            predicts.append(seg_predict_item)
            labels.append(label_item)

        tbar.set_postfix({"loss": loss.item()})
        tbar.update(1)

    predict_array = np.stack(predicts, axis=0)
    label_array = np.stack(labels, axis=0)
    dice = 2 * (predict_array * label_array).sum() / (predict_array.sum() + label_array.sum())
    recall = (predict_array[label_array == 1] == 1).sum() / (label_array == 1).sum()
    precision = (predict_array[label_array == 1] == 1).sum() / (predict_array == 1).sum()
    dice_list.append(dice)
    recall_list.append(recall)
    precision_list.append(precision)
    case_list.append(anchor_case_id)
    # print(anchor_case_id, case_id, dice)

    for index in range(len(case_list)):
        print("case_id:{}, dice:{}, recall:{}, precision:{}".format(
            case_list[index], round(dice_list[index],3), round(recall_list[index],3), round(precision_list[index], 3)))


    seg_dice = np.mean(np.array(dice_list))
    seg_recall = np.mean(np.array(recall_list))
    seg_precision = np.mean(np.array(precision_list))

    loss = np.mean(np.array(loss_list))
    return (loss, seg_dice, seg_recall, seg_precision)



def main_shell(batch_size=1, num_gpu=1, lr=0.001, max_epoll=100):

    torch.cuda.empty_cache()

    net = ExplainMode(input_channel=1, out_class=2)
    net.load_state_dict(torch.load("result/STS/pet_ct_epoll_5.pkl", map_location="cuda:0"))
    net = net.cuda(device=device_id)

    #net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )
    early_stopping = EarlyStopping(patience=30, verbose=True)

    pet_criterion = RecallTverskyLoss().cuda(device=device_id)
    ct_criterion = StandardSegLoss().cuda(device=device_id)
    fusion_crterion = StandardSegLoss().cuda(device=device_id)



    total_criterion = AutomaticWeightedLoss(3).cuda(device=device_id)
    criterion = [pet_criterion, ct_criterion, fusion_crterion, total_criterion]


    transform = MedicalTransformCompose(output_size=(512, 512), roi_error_range=15, use_roi=False)
    train_data = STSDataLoader(type="train", transform=transform)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_train_data = STSDataLoader(type="train")
    valid_train_data_loader = DataLoader(valid_train_data, batch_size=batch_size, shuffle=False)
    valid_data = STSDataLoader(type="valid")
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    best_dice = 0

    for epoll in range(max_epoll):
        epoch_str = f' Epoch {epoll + 1}/{max_epoll} '
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

        net.train()
        transform.train()
        torch.set_grad_enabled(True)
        train(data_loader=train_data_loader, net=net, criterion=criterion, optimizer=optimizer)

        net.eval()
        transform.eval()
        torch.set_grad_enabled(False)
        (train_loss, train_seg_dice, train_seg_recall, train_seg_precision) = evaluate(data_loader=valid_train_data_loader, net=net, criterion=criterion, type="train")
        (valid_loss, valid_seg_dice, valid_seg_recall, valid_seg_precision) = evaluate(data_loader=valid_data_loader, net=net, criterion=criterion, type="valid")
        scheduler.step(train_loss)
        early_stopping(valid_loss)

        print("Train loss : {}, Train Dice : {}".format(round(train_loss, 6), round(train_seg_dice, 3)))
        print("Valid loss : {}, Valid Dice : {}".format(round(valid_loss, 6), round(valid_seg_dice, 3)))

        with open("result/STS/log.log", "a+") as file:
          file.writelines("Epoll:{}, T_Loss: {}, T_S_Dice: {}, T_S_Recall: {}, T_S_Pre: {}, "
                          "V_Loss: {}, V_S_Dice: {}, V_S_Recall: {}, V_S_Pre:{},"
                          " \n".format(
              epoll+1, round(train_loss, 6), round(train_seg_dice, 3), round(train_seg_recall, 3), round(train_seg_precision, 3),

                       round(valid_loss, 6), round(valid_seg_dice, 3), round(valid_seg_recall, 3), round(valid_seg_precision, 3)
          ))

        if best_dice < valid_seg_dice:
            best_dice = valid_seg_dice
            torch.save(net.state_dict(), "result/STS/epoll_{}.pkl".format(epoll+1))

        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':
    main_shell(batch_size=2, num_gpu=2, lr=0.001)