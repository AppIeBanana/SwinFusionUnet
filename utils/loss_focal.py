import torch

class Focal_Loss():
    def __init__(self, weight=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)



# Create an instance of the Focal_Loss class
focal_loss = Focal_Loss(weight=0.25, gamma=2)

# Generate example inputs
batch_size = 10
num_classes = 5
height = 32
width = 32

preds = torch.randn(batch_size, num_classes, height, width)
labels = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()

# Forward pass
loss = focal_loss.forward(preds, labels)

print(loss)