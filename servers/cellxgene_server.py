from fastmcp import FastMCP
from typing import Optional, Union, List, Dict, Any
import cellxgene_census

# Initialize FastMCP
mcp = FastMCP(name="cellxgene_server")

def convert_to_list(lst):
    """
    Convert non-list elements to lists. Handles None.
    """
    temp = []
    for el in lst:
        if el is None:
            temp.append([])
        elif isinstance(el, str):
            temp.append([el])
        elif isinstance(el, list):
            temp.append(el)
        else:
            temp.append([el])
    return temp

def structured_response(data=None, message="success", status="success"):
    """
    Create a consistent response format for MCP tools.

    Args:
        data (Any): The actual data payload (list, dict, str, etc.)
        message (str): A short message about the result.
        status (str): Either 'success' or 'error'.

    Returns:
        dict: Standardized response object.
    """
    return {
        "status": status,
        "context": message,
        "content": data if data is not None else [],
    }

@mcp.tool()
async def get_dataset_h5ad_uri(dataset_id: str = "cb5efdb0-f91c-4cbd-9ad4-9d4fa41c572d",
                                census_version: str = "stable") -> Dict:
    """
    Return download URI for dataset_id from CELLxGENE Census.
    """
    try:
        response = cellxgene_census.get_source_h5ad_uri(str(dataset_id), census_version=census_version)
        return structured_response(response.get("uri"))
    except Exception as e:
        return structured_response([], str(e))

@mcp.tool()
async def cellxgene_dataset_summary(
    tissue: Optional[str] = None,
    max_rows: int = 3
) -> List[Dict[str, Any]]:
    """
    List available datasets in the CELLxGENE Census, filtered by tissue.
    """
    census_version: str = "latest"
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Read entire dataset metadata table
        df = census["census_info"]["datasets"].read().concat().to_pandas()
        # Apply filters
        if tissue:
            df = df[df["dataset_title"].str.contains(tissue, case=False, na=False)]
            #df["dataset_h5ad_uri"] = df["dataset_id"].map(lambda do: cellxgene_census.get_source_h5ad_uri(do)["uri"])
        # Return limited rows
        return df.head(max_rows).to_dict(orient="records")

async def cellxgene_query(
    species="homo_sapiens",
    column_names=[
        "dataset_id",
        "assay",
        "suspension_type",
        "sex",
        "tissue_general",
        "tissue",
        "cell_type",
    ],
    tissue=None,
    cell_type=None,
    development_stage=None,
    disease=None,
    sex=None,
    is_primary_data=True,
    dataset_id=None,
    tissue_general_ontology_term_id=None,
    tissue_general=None,
    assay_ontology_term_id=None,
    assay=None,
    cell_type_ontology_term_id=None,
    development_stage_ontology_term_id=None,
    disease_ontology_term_id=None,
    donor_id=None,
    self_reported_ethnicity_ontology_term_id=None,
    self_reported_ethnicity=None,
    sex_ontology_term_id=None,
    suspension_type=None,
    tissue_ontology_term_id=None,
    census_version="stable",
    verbose=True,
    out=None,
) -> Dict[str, Any]:
    """
    Query CELLxGENE Census metadata by filters. Returns metadata and dummy download links.
    """
    # Default fallback to prevent huge queries
    if not tissue:
        tissue = ["decidua basalis"]
    if not cell_type:
        cell_type = ["syncytiotrophoblast cell"]
    if not dataset_id:
        dataset_id = "f171db61-e57e-4535-a06a-35d8b6ef8f2b"

    # Mapping of field names to values
    filter_args = {
        "dataset_id": dataset_id,
        "tissue_general_ontology_term_id": tissue_general_ontology_term_id,
        "tissue_general": tissue_general,
        "assay_ontology_term_id": assay_ontology_term_id,
        "assay": assay,
        "cell_type_ontology_term_id": cell_type_ontology_term_id,
        "cell_type": cell_type,
        "development_stage_ontology_term_id": development_stage_ontology_term_id,
        "development_stage": development_stage,
        "disease_ontology_term_id": disease_ontology_term_id,
        "disease": disease,
        "donor_id": donor_id,
        "self_reported_ethnicity_ontology_term_id": self_reported_ethnicity_ontology_term_id,
        "self_reported_ethnicity": self_reported_ethnicity,
        "sex_ontology_term_id": sex_ontology_term_id,
        "sex": sex,
        "suspension_type": suspension_type,
        "tissue_ontology_term_id": tissue_ontology_term_id,
        "tissue": tissue,
    }

    # Convert all values to list
    filter_args = {k: v if isinstance(v, list) else [v] for k, v in filter_args.items() if v is not None}

    # Build value_filter string
    filters = []
    if is_primary_data:
        filters.append("is_primary_data == True")
    for k, v in filter_args.items():
        if v:
            quoted = [f'"{x}"' for x in v]
            filters.append(f"{k} in [{', '.join(quoted)}]")
    value_filter = " and ".join(filters) if filters else None

    if verbose:
        print("== CELLXGENE QUERY START ==")
        print(f"Species: {species}")
        print(f"Value Filter: {value_filter}")
        print(f"Columns: {column_names}")

    # Query cell metadata .get_obs(census, "homo_sapiens", value_filter="sex == 'unknown'")
    with cellxgene_census.open_soma(census_version=census_version) as census:

        obs_df = cellxgene_census.get_obs(
            census,
            species,
            value_filter=value_filter,
            column_names=column_names
        ).head(100)  # Limit to 100 rows to avoid timeout

    dataset_ids = sorted(obs_df["dataset_id"].dropna().unique().tolist())
    
    # Save if needed
    if out:
        obs_df.to_csv(out, index=False)

    return {
        "n_results": len(obs_df),
        "columns": obs_df.columns.tolist(),
        "dataset_ids": dataset_ids,
        "preview": obs_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
