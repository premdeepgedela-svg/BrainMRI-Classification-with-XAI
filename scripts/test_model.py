import torch

from src.models.cnn import BrainMRICNN


def main():

    model = BrainMRICNN()

    print("=" * 60)
    print(model)
    print("=" * 60)

    dummy = torch.randn(8, 1, 240, 240)

    output = model(dummy)

    print("Input Shape :", dummy.shape)
    print("Output Shape:", output.shape)


if __name__ == "__main__":
    main()