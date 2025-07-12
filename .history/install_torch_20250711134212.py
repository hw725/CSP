import subprocess
import sys
import platform

def install_pytorch():
    """PyTorch 안정적인 버전 설치"""
    
    # PyTorch 2.1.2 (LTS 버전) 사용
    torch_version = "2.1.2"
    torchvision_version = "0.16.2"
    torchaudio_version = "2.1.2"
    
    # CUDA 버전 확인
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # CUDA 12.1 사용 (더 안정적)
            index_url = "https://download.pytorch.org/whl/cu121"
            packages = [
                f"torch=={torch_version}+cu121",
                f"torchvision=={torchvision_version}+cu121", 
                f"torchaudio=={torchaudio_version}+cu121"
            ]
        else:
            # CUDA 없으면 CPU 버전
            index_url = "https://download.pytorch.org/whl/cpu"
            packages = [
                f"torch=={torch_version}+cpu",
                f"torchvision=={torchvision_version}+cpu",
                f"torchaudio=={torchaudio_version}+cpu"
            ]
    except FileNotFoundError:
        # CUDA 없으면 CPU 버전
        index_url = "https://download.pytorch.org/whl/cpu"
        packages = [
            f"torch=={torch_version}+cpu",
            f"torchvision=={torchvision_version}+cpu", 
            f"torchaudio=={torchaudio_version}+cpu"
        ]
    
    # 설치 명령어
    cmd = [sys.executable, "-m", "pip", "install"] + packages + ["--index-url", index_url]
    
    print(f"Installing PyTorch {torch_version}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("PyTorch installation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch: {e}")
        return False
    
    return True

if __name__ == "__main__":
    install_pytorch()