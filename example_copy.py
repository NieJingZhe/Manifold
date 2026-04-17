import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, global_mean_pool
from dataset.drugdataset import QM9Dataset
from models.pytorch_geometric.pna import PNAConvSimple
from manifold import prob_low_dim
class Net(torch.nn.Module):
    # 保留 data 参数（最小改动），尽管目前没用到它
    def __init__(self, deg):
        super(Net, self).__init__()

        self.node_emb = torch.nn.Linear(39, 80)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConvSimple(in_channels=80, out_channels=80, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(80))

        self.mlp = Sequential(Linear(80, 40), ReLU(), Linear(40, 20), ReLU(), Linear(20, 32))#32就是Zi，之后再改

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
    
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)
        x = self.mlp(x)
        Phat = prob_low_dim(x, 0.583, 1.334)    
        return Phat

class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.cfg = config
        dataset = QM9Dataset(name=self.cfg.dataset, root=self.cfg.data_root, args=args)
        self.train_set = dataset[dataset.train_index]
        self.valid_set = dataset[dataset.valid_index]
        self.test_set = dataset[dataset.test_index]

   
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.bs, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.cfg.bs, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.cfg.bs, shuffle=False)

        # 统一使用 self.device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute in-degree histogram over training data.
        deg = torch.zeros(10, dtype=torch.long)
        for data in self.train_set:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        # 修正：传入 Net 的参数与 Net.__init__ 匹配；并把 model 放到 self.device 上
        self.model = Net(deg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=3e-6)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20, min_lr=0.0001)

    def train(self, epoch):
        self.model.train()

        total_loss = 0
        for data in self.train_loader:
            # 修正：使用 self.device（原来写成 device，会报错）
            data = data.to(self.device)
            self.optimizer.zero_grad()
            Phat = self.model(data.x, data.edge_index, None, data.batch)
            Q = prob_low_dim(data.pos, 0.583, 1.334)
            loss = torch.nn.BCELoss()(Phat, Q)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            self.optimizer.step()
        # len(self.train_loader.dataset) 保持不变（DataLoader 有 dataset 属性）
        return total_loss / len(self.train_loader.dataset)


    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        list_pred = []
        list_labels = []
        
        for data in loader:
            data = data.to(self.device)
            Phat = self.model(data.x, data.edge_index, None, data.batch)
            Q = prob_low_dim(data.pos, 0.583, 1.334)
            # list_pred.append(out)
            # list_labels.append(data.pos)

        loss = torch.nn.BCELoss()(Phat, Q)
        return loss

    def forward(self):
        # 使用实例属性 self.best，并在判断时也用 self.best
        self.best = (1000, 1000)

        for epoch in range(1, 201):
            loss = self.train(epoch)
            # 修正：使用 self.valid_loader（原来写成 self.val_loader，会报错）
            val_loss = self.test(self.valid_loader)
            test_loss = self.test(self.test_loader)
            # ReduceLROnPlateau 接受数值
            self.scheduler.step(val_loss)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_loss:.4f}, '
                  f'Test: {test_loss:.4f}')
            # 修正：比较并更新 self.best
            if val_loss < self.best[0]:
                self.best = (val_loss, test_loss)

        print(f'Best epoch val: {self.best[0]:.4f}, test: {self.best[1]:.4f}')




if __name__ == "__main__":
    import argparse

    # 1. 构造命令行参数或配置
    class Config:
        dataset = "qm9"
        data_root = "/home/liyong/data/processed_data"
        bs = 128
        device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    args = parser.parse_args()  # 如果你有命令行参数可以传

    # 2. 创建 Trainer
    trainer = Trainer(args=args, config=Config)

    # 3. 调用 forward 进行训练和测试
    trainer.forward()
