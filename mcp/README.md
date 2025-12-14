# CrystalFormer-CSP MCP Server

Model Context Protocol (MCP) server for Crystal Structure Prediction, enabling AI assistants to generate crystal structures.

**Note:** Ensure that the LLM has the ability to follow instructions and give correct parameters for the tools.

## Setup

### 1. Install Dependencies
```bash
python -m pip install fastmcp pandas python-dotenv
```

### 2. Configure MCP Client

Create or edit `~/.cursor/mcp.json` or `~/claude_desktop_config.json` with the following configuration:

```json
{
  "mcpServers": {
    "csp_mcp": {
      "command": "python",
      "args": ["/path/to/crystalformer/mcp/csp_mcp.py"],
    }
  }
}
```

### 3. Edit the `.env` file

Edit the `.env` file with the following variables:
```
RESTORE_PATH=/path/to/model/checkpoint
MODEL_PATH=/path/to/mlff_model.ckpt
CONVEX_HULL_PATH=/path/to/convex_hull_pbe.json.bz2
SAVE_PATH=/path/to/output
```

## Available Tools

### **generate_structures** - Crystal Structure Prediction
Main tool for generating crystal structures with complete postprocessing pipeline.

**Key Parameters:**
- `formula` (required): Chemical formula (e.g., 'H2O', 'TiO2')

**Returns:**
- `success`: bool, True if the structure generation was successful
- `output`: stdout/stderr from the postprocess.sh script
- `formula`: the chemical formula used for the structure generation
- `output_path`: the path to the output directory
- `best_cif`: dictionary containing the best CIF structure and metadata
- `error`: error message if the structure generation failed

## Usage Examples

### Basic Crystal Generation
After setting up the MCP server, you can use the following prompt in your LLM interface:

```
Generate H2O crystal structures
```

## Contributing

To add new tools or modify existing ones:

1. Add the tool function with proper `@mcp.tool` decorator
2. Use `Annotated` types for parameter descriptions
3. Ensure the tool returns LLM-friendly output (paths and summaries)
4. Test the tool with the MCP server
5. Update this README with the new tool documentation