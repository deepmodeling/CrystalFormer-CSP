"""
Crystal Structure Prediction MCP Server

This server provides a MCP interface for the Crystal Structure Prediction pipeline.

"""

import os
import asyncio
import shlex
import pandas as pd
from typing import Annotated

from dotenv import load_dotenv
from fastmcp import FastMCP

# Create the FastMCP server instance
mcp = FastMCP("CrystalFormer-CSP MCP Server")

# Load environment variables
load_dotenv(
    os.path.join(os.path.dirname(__file__),  ".env")
)

def _extract_best_cif_from_ehull(output_path: str | None, formula: str) -> dict:
    """
    Extract the CIF structure with the lowest ehull from relaxed_structures_{formula}_ehull.csv.
    
    Args:
        output_path: Path where the pipeline outputs are stored
        formula: Chemical formula used in the pipeline
    
    Returns:
        dict: Dictionary containing the best CIF structure and metadata
    """
    from pymatgen.core import Structure
    
    result = {
        "cif_content": "",
        "ehull": None,
        "metadata": {
            "formula": formula,
            "total_structures": 0
        },
        "error": None,
    }
    
    if not output_path:
        result["error"] = "Output path is unknown; set RESTORE_PATH or SAVE_PATH in environment."
        return result
    
    try:
        # Read the relaxed structures with ehull file
        ehull_file = os.path.join(output_path, f"relaxed_structures_{formula}_ehull.csv")
        if not os.path.exists(ehull_file):
            # Fallback to generic name
            alt_file = os.path.join(output_path, "relaxed_structures_ehull.csv")
            if os.path.exists(alt_file):
                ehull_file = alt_file
            else:
                result["error"] = f"File not found: {ehull_file} or {alt_file}"
                return result
        
        df_ehull = pd.read_csv(ehull_file)
        result["metadata"]["total_structures"] = len(df_ehull)
        
        if df_ehull.empty:
            result["error"] = f"No structures found in {ehull_file}"
            return result
        
        # Find the structure with the lowest ehull
        ehull_column = None
        for col in ("relaxed_ehull", "ehull", "unrelaxed_ehull"):
            if col in df_ehull.columns:
                ehull_column = col
                break
        
        if ehull_column is None:
            result["error"] = f"No ehull-like column found in {ehull_file}"
            return result
        
        # Get the index of the row with minimum ehull
        min_idx = df_ehull[ehull_column].idxmin()
        best_row = df_ehull.loc[min_idx]
        
        # Extract the CIF and convert to CIF format
        cif_val = None
        for col in ("relaxed_cif", "cif", "structure"):
            if col in df_ehull.columns:
                cif_val = best_row[col]
                break
        
        if cif_val is not None:
            # Convert to CIF format: try dict first, then CIF string
            try:
                struct = Structure.from_dict(eval(str(cif_val)))
                result["cif_content"] = struct.to(fmt="cif")
            except:
                try:
                    struct = Structure.from_str(str(cif_val), fmt="cif")
                    result["cif_content"] = struct.to(fmt="cif")
                except:
                    # If already CIF text, use as is
                    result["cif_content"] = str(cif_val)
        
        result["metadata"]["ehull"] = float(best_row[ehull_column])
        
        # Include additional info if available
        for field in ("energy", "initial_energy", "final_energy"):
            if field in df_ehull.columns:
                try:
                    result["metadata"][field] = float(best_row[field])
                except:
                    result["metadata"][field] = best_row[field]
                    
    except Exception as e:
        result["error"] = f"Error reading CIF file: {str(e)}"
    
    return result

@mcp.tool(
    "generate_structures",
    description=(
        "Crystal Structure Prediction pipeline for a given chemical formula. "
        "Runs postprocess.sh with the given chemical formula. "
        "Other parameters are controlled via environment variables "
        "(RESTORE_PATH, MODEL_PATH, CONVEX_HULL_PATH, SAVE_PATH) "
        "or postprocess.sh defaults."
    ),
)
async def generate_structures(
    formula: Annotated[str, "The chemical formula, e.g. 'H2O', 'TiO2'"],
) -> dict:
    """
    Run the CrystalGPT postprocessing pipeline (postprocess.sh) for a given formula.

    Parameters
    ----------
    formula:
        Chemical formula of the compound, e.g. 'H2O', 'TiO2'.

    Returns
    -------
    dict
        {
          'success': bool,
          'output': combined stdout/stderr from postprocess.sh,
          'formula': str,
          'output_path': str | None,
          'best_cif': {
              'cif_content': str,
              'ehull': float | None,
              'metadata': {...},
              'error': str | None
          },
          'error': str | None   # high-level error if the script failed
        }
    """
    import logging

    logger = logging.getLogger(__name__)

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "postprocess.sh")
    )

    if not os.path.isfile(script_path):
        msg = f"postprocess.sh not found at {script_path}"
        logger.error(msg)
        return {
            "success": False,
            "output": "",
            "formula": formula,
            "output_path": None,
            "best_cif": {
                "cif_content": "",
                "ehull": None,
                "metadata": {"formula": formula, "total_structures": 0},
                "error": msg,
            },
            "error": msg,
        }

    # Collect optional arguments from environment.
    restore_path_env = os.getenv("RESTORE_PATH")
    model_path_env = os.getenv("MODEL_PATH")
    convex_hull_env = os.getenv("CONVEX_HULL_PATH")
    save_path_env = os.getenv("SAVE_PATH")

    args = []

    if restore_path_env:
        args.extend(["-r", restore_path_env])
    if model_path_env:
        args.extend(["-m", model_path_env])
    if convex_hull_env:
        args.extend(["-c", convex_hull_env])
    if save_path_env:
        args.extend(["-s", save_path_env])

    # Formula is always provided by the tool caller.
    args.extend(["-f", formula])

    cmd = "bash " + shlex.quote(script_path) + " " + " ".join(
        shlex.quote(str(arg)) for arg in args
    )

    logger.info("Running postprocess.sh for formula %s", formula)

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    output = stdout.decode("utf-8", errors="replace")

    success = process.returncode == 0
    if not success:
        logger.error("postprocess.sh failed with return code %s", process.returncode)

    # Determine where outputs were written.
    output_path = (save_path_env or restore_path_env or "").rstrip("/ ")
    if not output_path:
        output_path = None

    best_cif = _extract_best_cif_from_ehull(output_path, formula)

    return {
        "success": success,
        "output": output,
        "formula": formula,
        "output_path": output_path,
        "best_cif": best_cif,
        "error": None if success else "postprocess.sh failed; see output for details.",
    }

if __name__ == "__main__":
    # Running this module directly will start the MCP server.
    mcp.run()