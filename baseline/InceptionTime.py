import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):

    def __init__(self, in_channels, n_filters=16, kernel_sizes=[5, 11], use_residual=False):
        super(InceptionModule, self).__init__()
        self.use_residual = use_residual

        # Two parallel convolutional layers with different kernel sizes
        self.conv_list = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                n_filters,
                kernel_size=k,
                padding=k // 2,
                bias=False
            ) for k in kernel_sizes
        ])

        # MaxPooling path
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        # Batch normalization
        total_filters = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_filters)

        # Residual connection (optional, disabled by default)
        if use_residual and in_channels != total_filters:
            self.residual = nn.Conv1d(in_channels, total_filters, kernel_size=1, bias=False)
        else:
            self.residual = None

    def forward(self, x):
        # x shape: (batch, channels, length)
        input_tensor = x

        # Parallel convolutional paths
        conv_outputs = [conv(x) for conv in self.conv_list]

        # MaxPooling path
        pool_out = self.maxpool(input_tensor)
        pool_out = self.conv_pool(pool_out)

        # Concatenate all paths
        x = torch.cat(conv_outputs + [pool_out], dim=1)
        x = self.bn(x)

        # Residual connection (if enabled)
        if self.use_residual and self.residual is not None:
            x = x + self.residual(input_tensor)

        x = F.relu(x)
        return x


class Model(nn.Module):
    """
    Simplified InceptionTime model for time series classification
    Reduced complexity with fewer modules and smaller filters
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name

        # Model parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # Number of input channels/features
        self.num_class = configs.num_class

        # Simplified parameters (reduced from original)
        self.n_filters = getattr(configs, 'n_filters', 16)  # Reduced from 32
        self.kernel_sizes = getattr(configs, 'kernel_sizes', [5, 11])  # Only 2 sizes instead of 3
        self.depth = getattr(configs, 'depth', 3)  # Reduced from 6
        self.use_residual = getattr(configs, 'use_residual', False)  # Disabled by default

        # Calculate output channels
        self.out_channels = self.n_filters * (len(self.kernel_sizes) + 1)

        # Build Inception modules
        self.inception_modules = nn.ModuleList()
        in_channels = self.enc_in

        for i in range(self.depth):
            self.inception_modules.append(
                InceptionModule(
                    in_channels=in_channels,
                    n_filters=self.n_filters,
                    kernel_sizes=self.kernel_sizes,
                    use_residual=self.use_residual
                )
            )
            in_channels = self.out_channels

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        if self.task_name == 'classification':
            self.classifier = nn.Linear(self.out_channels, self.num_class)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc shape: (batch, seq_len, features)
        # Transpose to (batch, features, seq_len)
        x = x_enc.transpose(1, 2)

        # Pass through Inception modules
        for inception_module in self.inception_modules:
            x = inception_module(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.squeeze(-1)

        # Classification
        if self.task_name == 'classification':
            output = self.classifier(x)
            return output
        else:
            return x