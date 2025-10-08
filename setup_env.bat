@echo off
echo ============================================
echo   🚀 Setting up phi3-rag-benchmark environment
echo ============================================

:: ---- 1️⃣ Check if conda is available ----
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda not found! Please install Miniconda or Anaconda first.
    pause
    exit /b
)

:: ---- 2️⃣ Create a new conda environment ----
set ENV_NAME=phi3-rag-benchmark
echo Creating Conda environment: %ENV_NAME%
conda create -y -n %ENV_NAME% python=3.10

:: ---- 3️⃣ Activate environment ----
call conda activate %ENV_NAME%

:: ---- 4️⃣ Install PyTorch (with CUDA 12.1) ----
echo Installing PyTorch with CUDA 12.1 support...
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

:: ---- 5️⃣ Install Transformers ----
echo Installing Transformers...
pip install transformers==4.56.2

:: ---- 6️⃣ Install all other dependencies ----
echo Installing project dependencies...
pip install -r requirements.txt

:: ---- 7️⃣ Verify installations ----
echo Checking installations...
python -c "import torch, transformers; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Transformers: {transformers.__version__}')"

echo ============================================
echo ✅ Environment setup complete!
echo Run: conda activate %ENV_NAME%
echo ============================================
pause
