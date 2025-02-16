import torch

# MPS 장치가 사용 가능한지 확인
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    # 텐서를 MPS 장치로 이동
    x = torch.randn(1, 3).to(mps_device)
    print('ok')
else:
    print("MPS 장치가 지원되지 않습니다.")
