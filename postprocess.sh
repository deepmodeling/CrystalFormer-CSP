
#!/bin/bash

# CrystalGPT Postprocessing Script
# This script runs the complete postprocessing pipeline for generated crystal structures

set -e  # Exit on any error

# Default configuration
DEFAULT_BASE_DATA_PATH="/home/user_wanglei/private/datafile/crystalgpt"
DEFAULT_RESTORE_PATH="${DEFAULT_BASE_DATA_PATH}/csp/csp-85ed6/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_4_H_8_k_32_m_64_e_32_drop_0.3_0.1/"
DEFAULT_MODEL_PATH="${DEFAULT_BASE_DATA_PATH}/checkpoint/alex20/orb-v3-conservative-inf-mpa-20250404.ckpt"
DEFAULT_CONVEX_HULL_PATH="${DEFAULT_BASE_DATA_PATH}/checkpoint/alex20/convex_hull_pbe.json.bz2"
DEFAULT_FORMULA="H2O"
DEFAULT_K="40"
DEFAULT_NUM_SAMPLES="100"
DEFAULT_BATCHSIZE="100"
DEFAULT_TEMPERATURE="1.0"
DEFAULT_NUM_IO_PROCESS="20"
DEFAULT_EPOCH="epoch_030000.pkl"
DEFAULT_NF="5"
DEFAULT_KX="16"
DEFAULT_KL="4"
DEFAULT_H0_SIZE="256"
DEFAULT_TRANSFORMER_LAYERS="16"
DEFAULT_NUM_HEADS="8"
DEFAULT_KEY_SIZE="32"
DEFAULT_MODEL_SIZE="256"
DEFAULT_EMBED_SIZE="256"
DEFAULT_RELAXATION="true"
DEFAULT_LABEL=""
DEFAULT_SPACEGROUP=""
DEFAULT_SAVE_PATH=""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --restore-path PATH     Base restore path (default: $DEFAULT_RESTORE_PATH)"
    echo "  -m, --model-path PATH       Model checkpoint path (default: $DEFAULT_MODEL_PATH)"
    echo "  -c, --convex-hull PATH      Convex hull path (default: $DEFAULT_CONVEX_HULL_PATH)"
    echo "  -f, --formula FORMULA       Formula (default: ${DEFAULT_FORMULA})"
    echo "  -t, --temperature TEMP      Temperature (default: $DEFAULT_TEMPERATURE)"
    echo "  -p, --num-io-process NUM    Number of IO processes (default: $DEFAULT_NUM_IO_PROCESS)"
    echo "  -e, --epoch EPOCH           Epoch file (default: $DEFAULT_EPOCH)"
    echo "  -k  --K                     Number of top spacegroups (default: $DEFAULT_K)"
    echo "  -n, --num_samples           Number of samples for each spacegroup (default: $DEFAULT_NUM_SAMPLES)"
    echo "  -b, --batchsize             Number of batchsize (default: $DEFAULT_BATCHSIZE)"
    echo "  --Nf NF                     Nf parameter (default: $DEFAULT_NF)"
    echo "  --Kx KX                     Kx parameter (default: $DEFAULT_KX)"
    echo "  --Kl KL                     Kl parameter (default: $DEFAULT_KL)"
    echo "  --h0_size H0_SIZE           H0 size parameter (default: $DEFAULT_H0_SIZE)"
    echo "  --transformer-layers LAYERS Transformer layers (default: $DEFAULT_TRANSFORMER_LAYERS)"
    echo "  --num_heads HEADS           Number of heads (default: $DEFAULT_NUM_HEADS)"
    echo "  --key_size SIZE             Key size (default: $DEFAULT_KEY_SIZE)"
    echo "  --model_size SIZE           Model size (default: $DEFAULT_MODEL_SIZE)"
    echo "  --embed_size SIZE           Embed size (default: $DEFAULT_EMBED_SIZE)"
    echo "  --relaxation RELAXATION     Enable/disable relaxation (default: $DEFAULT_RELAXATION)"
    echo "  --label LABEL               Label for the experiment (default: '')"
    echo "  --spacegroup ID             Spacegroup ID (default: None)"
    echo "  --save-path PATH            Directory to save samples and postprocessed outputs (default: RESTORE_PATH)"
    echo "  --skip-sample               Skip the sampling step"
    echo "  --skip-convert              Skip the structure conversion step"
    echo "  --skip-energy               Skip the energy computation step"
    echo "  --skip-ehull                Skip the e-hull computation step"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -r /path/to/restore -f H2O"
    echo "  $0 --relaxation false  # Disable relaxation"
    echo "  $0 --relaxation true   # Enable relaxation (default)"
}

# Parse command line arguments
RESTORE_PATH="$DEFAULT_RESTORE_PATH"
MODEL_PATH="$DEFAULT_MODEL_PATH"
CONVEX_HULL_PATH="$DEFAULT_CONVEX_HULL_PATH"
FORMULA="$DEFAULT_FORMULA"
NUM_SAMPLES="$DEFAULT_NUM_SAMPLES"
TEMPERATURE="$DEFAULT_TEMPERATURE"
NUM_IO_PROCESS="$DEFAULT_NUM_IO_PROCESS"
EPOCH="$DEFAULT_EPOCH"
K="$DEFAULT_K"
NUM_SAMPLES="$DEFAULT_SAMPLES"
BATCHSIZE="$DEFAULT_BATCHSIZE"
NF="$DEFAULT_NF"
KX="$DEFAULT_KX"
KL="$DEFAULT_KL"
H0_SIZE="$DEFAULT_H0_SIZE"
TRANSFORMER_LAYERS="$DEFAULT_TRANSFORMER_LAYERS"
NUM_HEADS="$DEFAULT_NUM_HEADS"
KEY_SIZE="$DEFAULT_KEY_SIZE"
MODEL_SIZE="$DEFAULT_MODEL_SIZE"
EMBED_SIZE="$DEFAULT_EMBED_SIZE"
RELAXATION="$DEFAULT_RELAXATION"
LABEL="$DEFAULT_LABEL"
SPACEGROUP="$DEFAULT_SPACEGROUP"
SAVE_PATH="$DEFAULT_SAVE_PATH"
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
        -s|--save-path)
            SAVE_PATH="$2"
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
        -f|--formula)
            FORMULA="$2"
            shift 2
            ;;
        -n|--num_samples)
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
        -e|--epoch)
            EPOCH="$2"
            shift 2
            ;;
        -k|--K)
            K="$2"
            shift 2
            ;;
        --Nf)
            NF="$2"
            shift 2
            ;;
        --Kx)
            KX="$2"
            shift 2
            ;;
        --Kl)
            KL="$2"
            shift 2
            ;;
        --h0_size)
            H0_SIZE="$2"
            shift 2
            ;;
        --transformer_layers)
            TRANSFORMER_LAYERS="$2"
            shift 2
            ;;
        --num_heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --key_size)
            KEY_SIZE="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --embed_size)
            EMBED_SIZE="$2"
            shift 2
            ;;
        --relaxation)
            RELAXATION="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --spacegroup)
            SPACEGROUP="$2"
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

# Determine save path for samples and postprocessed outputs
if [[ -z "$SAVE_PATH" ]]; then
    SAVE_PATH="$RESTORE_PATH"
fi

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
OUTPUT_STRUCT_FILE="output_${FORMULA}_struct.csv"
RELAXED_STRUCT_FILE="relaxed_structures_${FORMULA}.csv"

echo "=== CrystalGPT Postprocessing Pipeline ==="
echo "Restore path (checkpoint): $RESTORE_PATH"
echo "Save/output path: $SAVE_PATH"
echo "Model path: $MODEL_PATH"
echo "Convex hull path: $CONVEX_HULL_PATH"
echo "Formula: $FORMULA"
echo "Number of samples: $NUM_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Epoch file: $EPOCH"
echo "=========================================="

# Ensure save path exists
mkdir -p "$SAVE_PATH"

# Step 1: Sample structures
if [[ "$SKIP_SAMPLE" == false ]]; then
    echo ""
    echo "Step 1: Sampling structures..."
    
    # Build the command arguments
    SAMPLE_ARGS=(
        --optimizer none
        --restore_path "$EPOCH_PATH"
        --K "$K"
        --verbose 1
        --batchsize "$BATCHSIZE"
        --num_samples "$NUM_SAMPLES"
        --formula "$FORMULA"
        --temperature "$TEMPERATURE"
        --Nf "$NF"
        --Kx "$KX"
        --Kl "$KL"
        --h0_size "$H0_SIZE"
        --transformer_layers "$TRANSFORMER_LAYERS"
        --num_heads "$NUM_HEADS"
        --key_size "$KEY_SIZE"
        --model_size "$MODEL_SIZE"
        --embed_size "$EMBED_SIZE"
        --save_path "$SAVE_PATH"
    )
    
    # Add spacegroup argument if specified
    if [[ -n "$SPACEGROUP" ]]; then
        SAMPLE_ARGS+=(--spacegroup "$SPACEGROUP")
    fi
    
    python ./main.py "${SAMPLE_ARGS[@]}" 
    
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
        --output_path "$SAVE_PATH/" \
        --formula "$FORMULA" \
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

    # Build mlff_relax command arguments
    MLFF_ARGS=(
        --restore_path "$SAVE_PATH/"
        --filename "$OUTPUT_STRUCT_FILE"
        --model orb-v3-conservative-inf-mpa
        --label "$FORMULA"
        --max_natoms_per_batch 10000
        --model_path "$MODEL_PATH"
    )
    
    # Add relaxation flag if enabled
    if [[ "$RELAXATION" == "true" ]]; then
        MLFF_ARGS+=(--relaxation)
    fi
    
    python ./scripts/mlff_relax_batch.py "${MLFF_ARGS[@]}"
    
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

    # Build e_above_hull_alex command arguments
    EHULL_ARGS=(
        --convex_path "$CONVEX_HULL_PATH"
        --restore_path "$SAVE_PATH/"
        --filename "$RELAXED_STRUCT_FILE"
    )
    
    # Add label if provided
    if [[ -n "$LABEL" ]]; then
        EHULL_ARGS+=(--label "$LABEL")
    else
        EHULL_ARGS+=(--label "$FORMULA")
    fi
    
    # Add relaxation flag if enabled
    if [[ "$RELAXATION" == "true" ]]; then
        EHULL_ARGS+=(--relaxation)
    fi
    
    python ./scripts/e_above_hull_alex.py "${EHULL_ARGS[@]}" 
    
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
