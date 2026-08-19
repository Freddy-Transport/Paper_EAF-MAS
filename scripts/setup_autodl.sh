#!/bin/bash
# =============================================================
# AutoDL 环境初始化脚本
# 使用方式：在 AutoDL 终端中执行
#   bash scripts/setup_autodl.sh            # 默认（国内镜像加速）
#   bash scripts/setup_autodl.sh --no-mirror # 使用官方 PyPI
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "===== AutoDL Environment Setup ====="
echo "Project root: $PROJECT_ROOT"

# ---- 镜像选项（默认使用清华镜像，在 AutoDL 国内环境明显更快）----
USE_MIRROR=true
for arg in "$@"; do
    [ "$arg" = "--no-mirror" ] && USE_MIRROR=false
done

if $USE_MIRROR; then
    PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    echo "Using Tsinghua PyPI mirror for faster installs."
else
    PIP_MIRROR=""
    echo "Using official PyPI (no mirror)."
fi

pip_install() {
    # shellcheck disable=SC2086
    pip install $PIP_MIRROR "$@"
}

# ---- 检测 PyTorch / CUDA ----
echo ""
echo "[1/5] Checking PyTorch..."
if python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "  PyTorch already installed, skipping."
else
    echo "  Installing PyTorch (cu118)..."
    # 自动检测 CUDA 版本选择合适的 wheel
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | cut -d. -f1 || echo "11")
    if [ "$CUDA_VER" -ge 12 ] 2>/dev/null; then
        pip_install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        pip_install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
fi

# ---- 基础依赖 ----
echo ""
echo "[2/5] Installing base dependencies..."
pip_install transformers==4.33.3
pip_install huggingface-hub==0.24.0
pip_install "numpy==1.25.2"
pip_install scikit-learn
pip_install safetensors
pip_install tqdm
pip_install matplotlib

# ---- 多智能体系统依赖 ----
echo ""
echo "[3/5] Installing multi-agent dependencies..."
pip_install "pydantic>=2.0"
pip_install pandas
pip_install openpyxl
pip_install requests

# ---- RAG 和 LLM 相关依赖 ----
echo ""
echo "[4/5] Installing RAG & LLM dependencies..."
pip_install langchain
pip_install langchain-openai
pip_install langchain-community
pip_install faiss-cpu
pip_install sentence-transformers

# ---- 以可编辑模式安装项目本身 ----
echo ""
echo "[5/5] Installing project in editable mode..."
pip install -e .

# ---- 配置 HuggingFace 镜像（AutoDL 国内环境无法直连 HF）----
if $USE_MIRROR; then
    echo ""
    echo "Setting HuggingFace mirror (hf-mirror.com)..."
    export HF_ENDPOINT="https://hf-mirror.com"
    # 写入当前 shell 的 bashrc（仅追加一次）
    if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
        echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
        echo "  Added HF_ENDPOINT to ~/.bashrc"
    fi
fi

echo ""
echo "===== Setup Complete ====="
echo ""
echo "Next steps:"
echo "  1. Configure LLM API key (copy and fill in .env.example):"
echo "     cp .env.example .env && vi .env"
echo "     source .env"
echo ""
echo "  2. Verify GPU:"
echo "     python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\""
echo ""
echo "  3. Run prediction:"
echo "     bash scripts/run_on_autodl.sh"
