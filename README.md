# 环境部署

### 一、 Anaconda安装  
#### 1.软件下载[清华下载站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)  
![1](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/c93e80f0-6cff-48a3-9e1e-ad5505b92b89)  
#### 2.一路next  
![2](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/7f96a272-f6b9-4a67-be4e-0f2de720fc44)  
#### 3.勾选上（免得后期手动添加环境变量）  
![3](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/21709f19-8558-4f14-ab05-7579dde1680a)  
*注意：软件安装路径最好不要有中文或者空格（后期可能会引起很多奇怪的问题）*  
#### 4.虚拟环境创建，打开cmd，输入`conda create -n catmeows python=3.9`  
![4](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/b1314fdd-6be0-4bc1-8b27-60ce72ea3698)  
等待安装结束  
![5](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/afee767e-d851-4562-b9a2-da54cd4f3e90)  
### 二、 Pytorch安装  
#### 1.软件下载[官网](https://pytorch.org/)（尽量选择老的版本）  
![image](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/3e24fc72-e898-436f-83d7-760854417cda)  
打开英伟达控制面板，查看支持的最高的cuda版本  
![image](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/c309cffa-b30c-4fd0-bc38-eac286a4dde2)  
回到pytorch官网,根据自己的情况选择（A卡或轻薄本请选择CPU版本）  
![image](https://github.com/hemo528/CatMeows_recogintion/assets/40025914/97d2bcac-f56a-438b-b878-acf10ff9e469)  
#### 2.使用pip命令进行安装`pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`  
*安装需要一定时间，耐心等待*  
#### 3.检验是否安装成功  
在cmd输入  
```
python
>>> import torch  
>>> torch.ones(3).cuda()  
tensor([1., 1., 1.], device='cuda:0')  
```  
若输出`tensor([1., 1., 1.], device='cuda:0') `则pytorch安装成功  












