import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        """
        Multi-head self-attention module.

        Args:
            input_dim: Dimension of the input feature.
            num_heads: Number of attention heads.
        """

        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads."

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear layers for query, key, and value
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Output linear layer after concatenating heads
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-head self-attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, num_features, input_dim).

        Returns:
            Tensor after applying multi-head self-attention.
        """
        batch_size = x.size(0)

        # Linear projections for query, key, value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Reshape for multi-head attention: (batch_size, num_heads, num_features, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate heads and pass through the final linear layer
        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        )
        output = self.fc_out(attention_output)

        return output
