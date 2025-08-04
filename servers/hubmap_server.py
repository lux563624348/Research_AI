import httpx
from typing import Optional, List, Dict
from fastmcp import FastMCP
import pandas as pd

# Initialize FastMCP
mcp = FastMCP(name="hubmap-server")

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
        "message": message,
        "data": data if data is not None else [],
    }

def filter_by_organ(df, organ="heart"):
    return df[df['origin_samples_unique_mapped_organs'].str.contains(organ, case=False, na=False)]
    

def build_dataset_descriptions(df):
    """
    Build a list of dicts with HuBMAP dataset URLs and descriptions from selected columns.
    
    Args:
        df (pd.DataFrame): A DataFrame with columns including:
            - 'uuid', 'donor.hubmap_id', 'puck_id', 'pi',
              'origin_samples_unique_mapped_organs', 'analyte_class',
              'assay_type', 'assay_input_entity', 'spatial_target',
              'sc_isolation_entity', 'sc_isolation_tissue_dissociation'
    
    Returns:
        List[dict]: Each dict has keys 'url' and 'description'
    """
    results = []

    for _, row in df.iterrows():
        uuid = row.get('uuid', '')
    
        description_fields = [
            "Author: " + str(row.get('pi', '')),
            row.get('origin_samples_unique_mapped_organs', ''),
            row.get('analyte_class', ''),
            row.get('assay_type', ''),
            row.get('assay_input_entity', ''),
            row.get('spatial_target', ''),
            row.get('sc_isolation_entity', ''),
            row.get('sc_isolation_tissue_dissociation', ''),
            f"https://portal.hubmapconsortium.org/browse/dataset/{uuid}"
        ]
        # Filter out None or empty string, then join with " | "
        description = " | ".join(str(f) for f in description_fields if pd.notna(f) and str(f).strip())
        results.append(description)
    return results

@mcp.tool()
async def hubmap_dataset_summary(
    number: int = 5,
    author: str = True,
    organ: str = "heart",
) -> dict:
    """
    Search HuBMAP dataset metadata for a given organ and return summaries with URLs and descriptions.

    Args:
        df (pd.DataFrame): HuBMAP metadata dataframe, must include specific columns.
        organ (str): Substring to search in 'origin_samples_unique_mapped_organs'.
    
    Returns:
        List[Dict]: List of dataset descriptions with URL and summary.
    """
    Path_csv = "/home/xli_p14/github/Research_AI/servers/hubmap-datasets-metadata-2025-08-03_13-39-35.tsv"
    df_tem = pd.read_csv(Path_csv, sep='\t')

    if author:
        # 1. Remove rows where 'pi' is NaN
        df_tem = df_tem.dropna(subset=['pi'])

    filter_df = filter_by_organ(df_tem, organ).head(number)
    # Generate result
    data = build_dataset_descriptions(filter_df)

    return structured_response(data)


#base_url = "https://search.api.hubmapconsortium.org/v3"
#url = f"{base_url}/{endpoint}"
async def make_hubmap_endpoint_request(endpoint: str, query_param: dict) -> Optional[dict]:
    """
    Query HuBMAP API v3 and return the JSON response.

    Args:
        endpoint (str): HuBMAP API endpoint path (e.g., "search", "entities/{id}", etc.).
        query_param (dict): Dictionary of query parameters.

    Returns:
        Optional[dict]: Parsed JSON response from HuBMAP API, or None if an error occurred.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            base_url = "https://search.api.hubmapconsortium.org/v3"
            url = f"{base_url}/{endpoint.lstrip('/')}"
            response = await client.get(url, params=query_param)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"❌ HuBMAP API request failed: {e}")
        return None

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

async def get_hubmap_metadata_by_organ(
    organ: str,
    size: int = 50,
    verbose: bool = False
) -> List[Dict]:
    """
    Query HuBMAP metadata records filtered by organ (tissue_type).
    Returns a list of metadata records with basic dataset fields.
    """
    params = {
        "q": f"tissue_type:{organ}",
        "size": size,
        "from": 0,
        "sort": "last_modified_timestamp:desc"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(HUBMAP_API, params=params)
            response.raise_for_status()
            results = response.json()

            hits = results.get("hits", {}).get("hits", [])
            data = [hit.get("_source", {}) for hit in hits]

            if verbose:
                print(f"Found {len(data)} results for organ: {organ}")

            return structured_response(data)

    except Exception as e:
        return structured_response([], str(e))
    
async def search_metadata_by_organ(organ: str, size: int = 20) -> Optional[dict]:

    """
    Search HuBMAP metadata entries for a given organ using the HuBMAP API.

    Args:
        organ (str): Name of the organ to search for (e.g., "kidney").
        size (int): Maximum number of entries to return (default: 20).

    Returns:
        Optional[dict]: Search results containing metadata for the specified organ.
    """
    query = {
        "q": f"organ:{organ}",
        "size": size,
        "sort": "last_modified_timestamp:desc"
    }

    result = await make_hubmap_endpoint_request("search", query)
    
    if result is None or "hits" not in result:
        return {"status": "error", "message": "No results or request failed."}
    
    return {

        "status": "success",
        "count": len(result["hits"]["hits"]),
        "results": [
            {
                "entity_id": hit["_id"],
                "organ": hit["_source"].get("organ"),
                "donor_id": hit["_source"].get("donor", {}).get("hubmap_id"),
                "last_modified": hit["_source"].get("last_modified_timestamp"),
                "sample_type": hit["_source"].get("sample_category"),
            }
            for hit in result["hits"]["hits"]
        ]
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")