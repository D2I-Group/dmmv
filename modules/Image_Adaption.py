import math
import torch
from torch import nn
import torch.nn.functional as F


class Conv_Adaption(nn.Module):
    def __init__(self, c_in, output_size=244):
        super().__init__()
        self.conv_in1 = nn.Conv2d(c_in, c_in // 2, kernel_size=3, padding=1)
        self.conv_in2 = nn.Conv2d(c_in // 2, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c_in // 2)
        self.bn2 = nn.BatchNorm2d(3)
        self.output_size = output_size

    def forward(self, x):
        conv_out = self.conv_in1(x)
        conv_out = self.bn1(conv_out)
        conv_out = F.gelu(conv_out)
        conv_out = self.conv_in2(conv_out)
        conv_out = self.bn2(conv_out)
        conv_out = F.interpolate(
            conv_out,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return conv_out


class Self_attention_Adaption(nn.Module):
    def __init__(self, c_in, d_model=256, output_size=224):
        super().__init__()

        self.conv = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in)
        self.bn = nn.BatchNorm2d(c_in)
        self.key_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))
        self.output_size = output_size
        self.middle_size = int(math.sqrt(d_model))

        self.conv_out = nn.Conv2d(c_in, 3, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(3)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        conv_out = F.gelu(conv_out)
        conv_out = F.interpolate(
            conv_out,
            size=(self.middle_size, self.middle_size),
            mode="bilinear",
            align_corners=False,
        )
        conv_out = conv_out.flatten(2)

        # Generate keys and queries
        keys = self.key_proj(conv_out)
        queries = self.query_proj(conv_out)

        # Calculate attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale.to(
            queries.device
        )
        attention = torch.softmax(scores, dim=-1)
        # Apply attention to original input while preserving original spatial dimensions
        x_flat = x.flatten(2)  # (batch_size, channels, height*width)
        attended = torch.matmul(attention, x_flat)

        # Reshape back to original spatial dimensions
        attended = attended.view(batch_size, channels, height, width)
        output = self.bn_out(self.conv_out(attended))
        output = F.interpolate(
            output,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return output
