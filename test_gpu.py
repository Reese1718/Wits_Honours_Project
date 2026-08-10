import torch

print("===== GPU TEST =====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Actually perform a small operation on the GPU
    device = torch.device("cuda")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2

    print(f"Tensor device: {y.device}")
    print(f"Test result: {y}")
    print("===== GPU TEST PASSED =====")
else:
    print("===== GPU NOT AVAILABLE =====")