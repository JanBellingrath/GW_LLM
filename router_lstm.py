import torch
import torch.nn as nn
from torch.nn import LSTMCell
import torch.nn.functional as F

class RouterLSTM(nn.Module):
    """
    Stateful LSTM-based router that takes the entire sequence embeddings as a single flattened input.
    The hidden state is preserved across transformer layers but reset between sequences.
    
    The router processes the entire prefix at each step by:
    1. Flattening the sequence into a single vector
    2. Padding to max sequence length
    3. Running one step through stacked LSTMCells
    4. Preserving state across transformer layer iterations
    """
    def __init__(self,
                 input_dim: int,      # D: embedding dimension
                 hidden_dim: int,     # Size of LSTM hidden states
                 num_layers: int,     # Number of LSTM layers
                 output_dim: int,     # Number of transformer layers/experts to route between
                 block_size: int,     # Maximum sequence length
                 dropout: float = 0.0,
                 config = None):      # Config object for normalization options
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.config = config
        
        # Per-token LayerNorm on each D-dim embedding
        self.ln_token = nn.LayerNorm(input_dim)
        
        # Stack of LSTMCells - first layer takes flattened sequence, rest take hidden_dim
        self.cells = nn.ModuleList([
            LSTMCell(
                input_size=input_dim * block_size if layer_idx == 0 else hidden_dim,
                hidden_size=hidden_dim
            )
            for layer_idx in range(num_layers)
        ])
        
        # Dropout between LSTM layers and on output
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm on LSTM output
        self.ln_output = nn.LayerNorm(hidden_dim)
        
        # Classifier projecting to logits over transformer layers/experts
        self.classifier = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
        
        # Initialize states list with correct length before first reset
        self.states = [(None, None) for _ in range(self.num_layers)]
        
        # Initialize hidden states to None
        self.reset_states()

    def _init_weights(self):
        """Initialize LSTM and classifier weights"""
        for cell in self.cells:
            # Initialize LSTM weights
            for name, param in cell.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # Initialize forget gate bias to 1.0 as per best practices
                    # bias_ih and bias_hh are concatenated into one tensor
                    n = param.size(0)
                    start_idx = n // 4
                    end_idx = n // 2
                    forget_bias_idx = slice(start_idx, end_idx)  # Forget gate is second quarter
                    param.data.zero_()
                    param.data[forget_bias_idx] = 1.0

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def reset_states(self):
        """Reset hidden states between sequences by setting them to None"""
        # start each sequence with no hidden or cell memory
        self.states = [(None, None) for _ in range(self.num_layers)]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """
        Process entire sequence prefix as one flattened input through stateful LSTMCells.
        
        Args:
            x: (batch_size, seq_len, input_dim) - Current sequence prefix embeddings
            lengths: Optional, ignored since we pad to block_size
            
        Returns:
            logits: (batch_size, seq_len, num_layers) - Logits over transformer layer choices
        """
        batch_size, seq_len, D = x.size()
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"
        assert seq_len <= self.block_size, f"Sequence length {seq_len} exceeds block_size {self.block_size}"
        
        # Store original dtype for consistent return
        orig_dtype = x.dtype
        
        # 1) Pad sequence to block_size in the token dimension
        if seq_len < self.block_size:
            pad_tokens = x.new_zeros(batch_size, self.block_size - seq_len, D)  # (B, pad, D)
            x_padded = torch.cat([x, pad_tokens], dim=1)                        # (B, block_size, D)
        else:
            x_padded = x[:, :self.block_size, :]                                # truncate if needed

        # 2) Flatten tokens and apply per-token LayerNorm
        x_tok = x_padded.reshape(batch_size * self.block_size, D)              # (B·block_size, D)
        x_tok_norm = self.ln_token(x_tok.to(torch.float32)).to(orig_dtype)     # (B·block_size, D)

        # 3) Re-flatten back into the full vector
        x_full = x_tok_norm.reshape(batch_size, self.block_size * D)           # (B, block_size·D)
        x_full = self.dropout(x_full)
            
        # 3) Single time-step through LSTMCell stack
        h_in = x_full
        new_states = []
        for i, cell in enumerate(self.cells):
            h_prev, c_prev = self.states[i]
            if h_prev is not None:
                h_prev = h_prev.to(orig_dtype)
                c_prev = c_prev.to(orig_dtype)
                # Run one step with previous state
                h_new, c_new = cell(h_in, (h_prev, c_prev))
            else:
                # Let LSTMCell initialize state to zeros
                h_new, c_new = cell(h_in)
            h_new = self.dropout(h_new)
            # Don't detach states between iterations to allow BPTT
            new_states.append((h_new, c_new))
            h_in = h_new  # Feed to next layer
            
        # 4) Save states for next transformer layer
        self.states = new_states
        
        # 5) Project final hidden state to logits (use float32 for stability)
        # Optionally normalize LSTM output before classification
        h_final = h_new.to(torch.float32)
        if hasattr(self.config, 'normalize_router_logits') and self.config.normalize_router_logits:
            h_final = self.ln_output(h_final)
        logits = self.classifier(h_final)  # (batch_size, num_layers)
        
        # 6) Compute routing weights in full precision for numerical stability
        routing_weights = F.softmax(logits, dim=-1)  # Already in float32
        routing_weights = routing_weights.to(orig_dtype)
        
        # 7) Expand routing weights to match sequence length
        routing_weights = routing_weights.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, num_layers)
        
        return routing_weights 