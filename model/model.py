import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .generator import CaptionGenerator
from .panns import Cnn10

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DACAN(BaseModel):
    """
    Refer `Diverse Audio Captioning via Adversarial Training`
    """
    def __init__(
        self,
        input_dim:int,
        vocab_size:int,
        num_layers:int = None,
        num_heads:int = None,
        ff_factor:int = None,
        max_seq_len:int = None,
    ):
        self.cnn = Cnn10(
            32000, 1024, 320, 64, 50, 14000, 527
        )
        pretrained_state = torch.load('saved/models/Cnn10_mAP=0.380.pth')
        self.cnn.load_state_dict(pretrained_state, strict=False)
        self.gen = CaptionGenerator(
            input_dim,
            vocab_size,
            num_layers,
            num_heads,
            ff_factor,
            max_seq_len,
        )

    def forward(self, inputs, input_masks, test=False):
        input_features = self.cnn(inputs)['embedding']
        return self.gen(input_features, input_masks, test=test)