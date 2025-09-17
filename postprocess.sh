
#!/bin/bash

# CrystalGPT Postprocessing Script
# This script runs the complete postprocessing pipeline for generated crystal structures

set -e  # Exit on any error

# Default configuration
DEFAULT_BASE_DATA_PATH="/home/user_wanglei/private/datafile/crystalgpt"
DEFAULT_RESTORE_PATH="${DEFAULT_BASE_DATA_PATH}/firsttry-Si-167/csp-17274/ppo_5_beta_0_adam_bs_500_lr_1e-05_decay_0_clip_1_A_119_W_28_N_21_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_16_k_64_m_64_e_32_drop_0.1"
DEFAULT_MODEL_PATH="${DEFAULT_BASE_DATA_PATH}/checkpoint/alex20/orb-v2-20241011.ckpt"
DEFAULT_CONVEX_HULL_PATH="${DEFAULT_BASE_DATA_PATH}/checkpoint/alex20/convex_hull_pbe_2023.12.29.json.bz2"
DEFAULT_ELEMENTS="Si"
DEFAULT_SPACEGROUP="167"
DEFAULT_NUM_SAMPLES="1000"
DEFAULT_BATCHSIZE="1000"
DEFAULT_TEMPERATURE="1.0"
DEFAULT_NUM_IO_PROCESS="20"
DEFAULT_EPOCH="epoch_005515.pkl"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --restore-path PATH     Base restore path (default: $DEFAULT_RESTORE_PATH)"
    echo "  -m, --model-path PATH       Model checkpoint path (default: $DEFAULT_MODEL_PATH)"
    echo "  -c, --convex-hull PATH      Convex hull path (default: $DEFAULT_CONVEX_HULL_PATH)"
    echo "  -e, --elements ELEMENTS     Elements (default: $DEFAULT_ELEMENTS)"
    echo "  -s, --spacegroup GROUP      Space group (default: $DEFAULT_SPACEGROUP)"
    echo "  -n, --num-samples NUM       Number of samples (default: $DEFAULT_NUM_SAMPLES)"
    echo "  -b, --batchsize SIZE        Batch size (default: $DEFAULT_BATCHSIZE)"
    echo "  -t, --temperature TEMP      Temperature (default: $DEFAULT_TEMPERATURE)"
    echo "  -p, --num-io-process NUM    Number of IO processes (default: $DEFAULT_NUM_IO_PROCESS)"
    echo "  -k, --epoch EPOCH           Epoch file (default: $DEFAULT_EPOCH)"
    echo "  --skip-sample               Skip the sampling step"
    echo "  --skip-convert              Skip the structure conversion step"
    echo "  --skip-energy               Skip the energy computation step"
    echo "  --skip-ehull                Skip the e-hull computation step"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -r /path/to/restore -e Si -s 167"
}

# Parse command line arguments
RESTORE_PATH="$DEFAULT_RESTORE_PATH"
MODEL_PATH="$DEFAULT_MODEL_PATH"
CONVEX_HULL_PATH="$DEFAULT_CONVEX_HULL_PATH"
ELEMENTS="$DEFAULT_ELEMENTS"
SPACEGROUP="$DEFAULT_SPACEGROUP"
NUM_SAMPLES="$DEFAULT_NUM_SAMPLES"
BATCHSIZE="$DEFAULT_BATCHSIZE"
TEMPERATURE="$DEFAULT_TEMPERATURE"
NUM_IO_PROCESS="$DEFAULT_NUM_IO_PROCESS"
EPOCH="$DEFAULT_EPOCH"
SKIP_SAMPLE=false
SKIP_CONVERT=false
SKIP_ENERGY=false
SKIP_EHULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--restore-path)
            RESTORE_PATH="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--convex-hull)
            CONVEX_HULL_PATH="$2"
            shift 2
            ;;
        -e|--elements)
            ELEMENTS="$2"
            shift 2
            ;;
        -s|--spacegroup)
            SPACEGROUP="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -b|--batchsize)
            BATCHSIZE="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -p|--num-io-process)
            NUM_IO_PROCESS="$2"
            shift 2
            ;;
        -k|--epoch)
            EPOCH="$2"
            shift 2
            ;;
        --skip-sample)
            SKIP_SAMPLE=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-energy)
            SKIP_ENERGY=true
            shift
            ;;
        --skip-ehull)
            SKIP_EHULL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate paths
if [[ ! -d "$RESTORE_PATH" ]]; then
    echo "Error: Restore path does not exist: $RESTORE_PATH"
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$CONVEX_HULL_PATH" ]]; then
    echo "Error: Convex hull path does not exist: $CONVEX_HULL_PATH"
    exit 1
fi

# Construct full paths
EPOCH_PATH="${RESTORE_PATH}/${EPOCH}"
OUTPUT_STRUCT_FILE="output_${SPACEGROUP}_struct.csv"
RELAXED_STRUCT_FILE="relaxed_structures.csv"

echo "=== CrystalGPT Postprocessing Pipeline ==="
echo "Restore path: $RESTORE_PATH"
echo "Model path: $MODEL_PATH"
echo "Convex hull path: $CONVEX_HULL_PATH"
echo "Elements: $ELEMENTS"
echo "Space group: $SPACEGROUP"
echo "Number of samples: $NUM_SAMPLES"
echo "Batch size: $BATCHSIZE"
echo "Temperature: $TEMPERATURE"
echo "Epoch file: $EPOCH"
echo "=========================================="

# Step 1: Sample structures
if [[ "$SKIP_SAMPLE" == false ]]; then
    echo ""
    echo "Step 1: Sampling structures..."
    python ./main.py \
        --optimizer none \
        --restore_path "$EPOCH_PATH" \
        --elements "$ELEMENTS" \
        --spacegroup "$SPACEGROUP" \
        --num_samples "$NUM_SAMPLES" \
        --batchsize "$BATCHSIZE" \
        --temperature "$TEMPERATURE"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Sampling failed"
        exit 1
    fi
    echo "Sampling completed successfully"
else
    echo "Skipping sampling step"
fi

# Step 2: Convert structures
if [[ "$SKIP_CONVERT" == false ]]; then
    echo ""
    echo "Step 2: Converting structures..."
    python ./scripts/awl2struct.py \
        --output_path "$RESTORE_PATH/" \
        --label "$SPACEGROUP" \
        --num_io_process "$NUM_IO_PROCESS"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Structure conversion failed"
        exit 1
    fi
    echo "Structure conversion completed successfully"
else
    echo "Skipping structure conversion step"
fi

# Step 3: Compute energy
if [[ "$SKIP_ENERGY" == false ]]; then
    echo ""
    echo "Step 3: Computing energy..."
    python ./scripts/mlff_relax.py \
        --restore_path "$RESTORE_PATH/" \
        --filename "$OUTPUT_STRUCT_FILE" \
        --model orb \
        --model_path "$MODEL_PATH"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Energy computation failed"
        exit 1
    fi
    echo "Energy computation completed successfully"
else
    echo "Skipping energy computation step"
fi

# Step 4: Compute e-hull
if [[ "$SKIP_EHULL" == false ]]; then
    echo ""
    echo "Step 4: Computing e-hull..."
    python ./scripts/e_above_hull_alex.py \
        --convex_path "$CONVEX_HULL_PATH" \
        --restore_path "$RESTORE_PATH/" \
        --filename "$RELAXED_STRUCT_FILE" \
        --label "$SPACEGROUP"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: E-hull computation failed"
        exit 1
    fi
    echo "E-hull computation completed successfully"
else
    echo "Skipping e-hull computation step"
fi

echo ""
echo "=== Postprocessing pipeline completed successfully! ==="
