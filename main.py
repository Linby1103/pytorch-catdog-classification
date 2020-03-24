from config import DefaultConfig
import torch as t
from torch.utils.data import DataLoader
from data import DogCat
import models
from torch.autograd import Variable
import os
from torchnet import meter
from models.ResNet34 import ResNet34
from utils.visualize import Visualizer

opt=DefaultConfig()

def train(**kwargs):
    opt.parse(kwargs)
#模型
    model=getattr(models,opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
#数据
    train_data=DogCat(opt.train_data_root,train=True)
    val_data=DogCat(opt.train_data_root,train=False)
    train_dataloader=DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)


    val_dataloader=DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)

#定义优化器和优化目标
    criterion=t.nn.CrossEntropyLoss()
    lr=opt.lr
    optimizer=t.optim.Adam(model.parameters(),lr=lr,weight_decay=opt.weight_decay)

    #统计指标
    loss_meter=meter.AverageValueMeter()
    confusion_matrix=meter.ConfusionMeter(2)

    previous_loss = 1e100

    #训练

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in enumerate(train_dataloader):
            input=Variable(data)
            target=Variable(label)
            if opt.use_gpu:
                input=input.cuda()
                target=target.cuda()
            optimizer.zero_grad()
            score=model(input)
            #print('score.dtype=',score.dtype," target.dtype=",target.dtype)
            loss =criterion(score,target)
            loss.backward()
            optimizer.step()

            #统计更新并可视化
            loss_meter.add(loss.item())
            runing_loss=loss.item()
            confusion_matrix.add(score.data,target.data)
            print('[%d ,%5d] loss=%.3f' % (epoch + 1, ii + 1, runing_loss))
            # if ii % opt.print_freq ==opt.print_freq-1:
            #     print('[%d ,%5d] loss=%.3f' %(epoch+1,ii+1,runing_loss/opt.print_freq))
                #vis.plot('loss',loss_meter.value()[0])

            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
        print("***finished %d epoch***\n" %epoch)
        model.save(counter=epoch)
        val_cm,val_accuracy=val(model,val_dataloader)
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            print('learn:%.4f\n' %lr )
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:

                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]



def val(model,dataloader):
    """计算模型在验证集上的准确率"""
    model.eval()
    confusion_matrix=meter.ConfusionMeter(2)#统计分类问题中的分类情况
    for ii,data in enumerate(dataloader):
        input ,label=data
        val_input=Variable(input,volatile=True)#?
        val_label=Variable(label.type(t.LongTensor),volatile=True)

        if opt.use_gpu:
            val_input=val_input.cuda()
            val_label=val_label.cuda()
        score=model(val_input)
        confusion_matrix.add(score.data.squeeze(),label.type(t.LongTensor))

        #模型回复为训练模式
        model.train()
        cm_value=confusion_matrix.value()
        accuracy=100.*(cm_value[0][0]+cm_value[1][1])/(cm_value.sum())

        return confusion_matrix,accuracy
def write_csv(results,file_name):
    import csv
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
def test(**kwargs):
    opt.parse(kwargs)

    #模型
    model=getattr(models,opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)

    if opt.use_gpu:
        model=opt.use_gpu
    train_data=DogCat(opt.test_data_root,test=True)
    test_dataloader=DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)

    result=[]
    for ii ,(data,path) in enumerate(test_dataloader):
        input=Variable(data,volatile=True)
        if opt.use_gpu:
            input=input.cuda()
        score = model(input)

        probability = t.nn.functional.softmax(score)[:, 1].data.tolist()
        batch_result = [(path_, _probability) for path_, _probability in zip(path, probability)]
        result += batch_result
        write_csv(result, opt.result_file)
        return result


# 创建一个txt文件，文件名为mytxtfile,并向文件写入msg
def text_create(msg):
    desktop_path = "D:/workspace/code/pytorch/DogCat/models/checkpoints/AlexNet_3.pth"  # 新创建的txt文件的存放路径
    file = open(desktop_path, 'w')
    file.write(msg)  # msg也就是下面的Hello world!
    file.close()



# 调用函数创建一个名为mytxtfile的.txt文件，并向其写入Hello world!
if __name__=='__main__':
    # text_create('hello')
    train()
    #test()






