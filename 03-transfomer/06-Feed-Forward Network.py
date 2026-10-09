import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=8,dff=8):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.ffnet = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )
    def forward(self, x):
        return self.ffnet(x)

def main():
    model = FeedForwardNetwork(d_model=8, dff=8)
    x = torch.randn(1, 2, 8)
    y = model(x)
    print("Input tensor",x)
    print("Output tensor",y)

if __name__ == "__main__":
    main()