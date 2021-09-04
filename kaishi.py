import deepmatcher as dm
import torch
from deepmatcher.data import MatchingDataset, MatchingIterator
import torch.nn as nn
import torch.nn.functional as F
import time
# torch.set_printoptions(profile="full")

def attribut(fields,embeddings):
	batch_list = torch.zeros(1,300).to('cuda')
	for field in fields:



				# Get token embedding
		value_embedding = embeddings[field]
	#     print('valuee',field,value_embedding)
		h = 0
		# print(field,value_embedding)
		for i in value_embedding.data:
			d = value_embedding.lengths

			c = torch.zeros(300).to('cuda')
					# for j in i:
					#     # print(j)
			if (int(d[h]) != 2):
				
				for j in range(1,int(d[h])-1):
					c.add_(i[j])
			
				c = torch.div(c,d[h]-2)         #做平均
				# print(c)

				c = torch.unsqueeze(c, 0).to('cuda')              #增加一维
				batch_list = torch.cat((batch_list,c),dim=0)
			h = h+1	
	return batch_list[1:]



def getbatch(input_batch,model,train):
	embeddings = {}
	for name in model.meta.all_text_fields:
			
		attr_input = getattr(input_batch, name)
				
		embeddings[name] = model.embed[name](attr_input)
				# print(embeddings['left_title'].data.shape)


	fields = train.all_text_fields
	# for field in fields:
	# 	value = getattr(input_batch, field)
	# 	print('value',field,value)
	batch_tensor = attribut(fields,embeddings)
	return batch_tensor


train, validation, test = dm.data.process(path='/home/learn/VAE/quan-vae-dataset/Beer',
    train='train.csv', validation='valid.csv', test='test.csv')




run_iter = MatchingIterator(
                train,
                train,
                train=True,
                batch_size=2,
                device='cuda',
                sort_in_buckets=True)


model = dm.MatchingModel(attr_summarizer='rnn')
model.run_train(train )
model.to('cuda')
# for batch_idx, batch in enumerate(run_iter):






 
# 超参数设置
# Hyper-parameters
num_epochs = 15
learning_rate = 1e-3

 
# 定义VAE类
# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=300, h_dim=200, z_dim=100):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
 
    # 编码  学习高斯分布均值与方差
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
 
    # 将高斯分布均值与方差参数重表示，生成隐变量z  若x~N(mu, var*var)分布,则(x-mu)/var=z~N(0, 1)分布
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    # 解码隐变量z
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
 
    # 计算重构值和隐变量z的分布参数
    def forward(self, x):
        mu, log_var = self.encode(x)# 从原始样本x中学习隐变量z的分布，即学习服从高斯分布均值与方差
        z = self.reparameterize(mu, log_var)# 将高斯分布均值与方差参数重表示，生成隐变量z
        x_reconst = self.decode(z)# 解码隐变量z，生成重构x’
        return x_reconst, mu, log_var# 返回重构值和隐变量的分布参数
 
# 构造VAE实例对象
model_vae = VAE().to('cuda')


 
# 选择优化器，并传入VAE模型参数和学习率
optimizer = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)
#开始训练
for epoch in range(num_epochs):
	datatime=0
	runtime=0
	batch_end=time.time()
	for i,x in enumerate(run_iter):
		batch_start = time.time()
        # datatime += batch_start - batch_end
		x = getbatch(x,model,train)
		# print(x.shape)
        # 前向传播
        #x = x.to('cuda').view(-1, image_size)# 将batch_size*1*28*28 ---->batch_size*image_size  其中，image_size=1*28*28=784
		x_reconst, mu, log_var = model_vae(x)# 将batch_size*748的x输入模型进行前向传播计算,重构值和服从高斯分布的隐变量z的分布参数（均值和方差）
 
        # 计算重构损失和KL散度
        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        # 重构损失
		reconst_loss = F.mse_loss(x_reconst, x, size_average=False)
        # KL散度
		kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
 
        # 反向传播与优化
        # 计算误差(重构误差和KL散度值)
		loss = reconst_loss + kl_div
        # 清空上一步的残余更新参数值
		optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
		loss.backward()
        # 将参数更新值施加到VAE model的parameters上
		optimizer.step()
        # batch_end = time.time()
        # runtime += batch_end - batch_start
        # 每迭代一定步骤，打印结果值
		if (i + 1) % 10 == 0:
			print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(epoch + 1, num_epochs, i + 1, len(run_iter), reconst_loss.item(), kl_div.item()))
        # print(datatime,runtime)
 
    # with torch.no_grad():
    #     # Save the sampled images
    #     # 保存采样值
    #     # 生成随机数 z
    #     z = torch.randn(batch_size, z_dim).to(device)# z的大小为batch_size * z_dim = 128*20
    #     # 对随机数 z 进行解码decode输出
    #     out = model.decode(z).view(-1, 1, 28, 28)
    #     # 保存结果值
    #     save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))
 
    #     # Save the reconstructed images
    #     # 保存重构值
    #     # 将batch_size*748的x输入模型进行前向传播计算，获取重构值out
    #     out, _, _ = model(x)
    #     # 将输入与输出拼接在一起输出保存  batch_size*1*28*（28+28）=batch_size*1*28*56
    #     x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
    #     save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))
# model.run_eval(test)
torch.save(model_vae.state_dict(),'/home/learn/VAE/ab.pth')
