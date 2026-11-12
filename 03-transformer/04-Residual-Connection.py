import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, d_model=8):
        super().__init__()


        self.linear = nn.Linear(
            d_model, d_model
        )

    def forward(self, x):
        return x + self.linear(x), self.linear(x)

def main():
    d_model = 8
    model = ResidualBlock(d_model)
    x = torch.randn(1,4,d_model)
    output, weight = model(x)
    print("Input tensor:", x)
    print("Output tensor:", output, weight)

if __name__ == "__main__":
    main()