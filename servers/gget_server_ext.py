#!/usr/bin/env python3
"""gget MCP Server - Bioinformatics query interface using the gget library."""

import os
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pathlib import Path
import uuid
import json

from fastmcp import FastMCP
from eliot import start_action
import gget

class TransportType(str, Enum):
    STDIO = "stdio"
    STDIO_LOCAL = "stdio-local"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3002"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # Changed default to stdio

# Typehints for common return patterns discovered in battle tests
SequenceResult = Union[Dict[str, str], List[str], str]
StructureResult = Union[Dict[str, Any], str]
SearchResult = Dict[str, Any]
LocalFileResult = Dict[Literal["path", "format", "success", "error"], Any]

class GgetMCPExtended(FastMCP):
    """gget MCP Server with bioinformatics tools."""
    
    def __init__(
        self, 
        name: str = "gget MCP Server",
        prefix: str = "gget_",
        transport_mode: str = "stdio",
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the gget tools with FastMCP functionality."""
        super().__init__(name=name, **kwargs)
        
        self.prefix = prefix
        self.transport_mode = transport_mode
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "gget_output"
        
        # Create output directory if in local mode
        if self.transport_mode == "stdio-local":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self._register_gget_tools()
    
    def _save_to_local_file(
        self, 
        data: Any, 
        format_type: str, 
        output_path: Optional[str] = None,
        default_prefix: str = "gget_output"
    ) -> LocalFileResult:
        """Helper function to save data to local files.
        
        Args:
            data: The data to save
            format_type: File format ('fasta', 'afa', 'pdb', 'json', etc.)
            output_path: Full output path (absolute or relative) or None to auto-generate
            default_prefix: Prefix for auto-generated filenames
            
        Returns:
            LocalFileResult: Contains path, format, success status, and optional error information
        """
        # Map format types to file extensions
        format_extensions = {
            'fasta': '.fasta',
            'afa': '.afa',
            'pdb': '.pdb',
            'json': '.json',
            'txt': '.txt',
            'tsv': '.tsv'
        }
        
        extension = format_extensions.get(format_type, '.txt')
        
        if output_path is None:
            # Generate a unique filename in the default output directory
            base_name = f"{default_prefix}_{str(uuid.uuid4())[:8]}"
            file_path = self.output_dir / f"{base_name}{extension}"
        else:
            # Use the provided path
            path_obj = Path(output_path)
            if path_obj.is_absolute():
                # Absolute path - use as is, but ensure it has the right extension
                if path_obj.suffix != extension:
                    file_path = path_obj.with_suffix(extension)
                else:
                    file_path = path_obj
            else:
                # Relative path - concatenate with output directory
                if not str(output_path).endswith(extension):
                    file_path = self.output_dir / f"{output_path}{extension}"
                else:
                    file_path = self.output_dir / output_path
        
        try:
            if format_type in ['fasta', 'afa']:
                self._write_fasta_file(data, file_path)
            elif format_type == 'pdb':
                self._write_pdb_file(data, file_path)
            elif format_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                # Default to text format
                with open(file_path, 'w') as f:
                    if isinstance(data, dict):
                        json.dump(data, f, indent=2, default=str)
                    else:
                        f.write(str(data))
                        
            return {
                "path": str(file_path),
                "format": format_type,
                "success": True
            }
        except Exception as e:
            return {
                "path": None,
                "format": format_type,
                "success": False,
                "error": str(e)
            }
    
    def _write_fasta_file(self, data: Any, file_path: Path) -> None:
        """Write sequence data in FASTA format.
        
        Handles multiple data formats discovered in battle tests:
        - Dict[str, str]: sequence_id -> sequence
        - List[str]: [header, sequence, header, sequence, ...]
        - str: raw data
        """
        with open(file_path, 'w') as f:
            if isinstance(data, dict):
                for seq_id, sequence in data.items():
                    f.write(f">{seq_id}\n")
                    # Write sequence with line breaks every 80 characters
                    for i in range(0, len(sequence), 80):
                        f.write(f"{sequence[i:i+80]}\n")
            elif isinstance(data, list):
                # Handle FASTA list format from gget.seq
                for i in range(0, len(data), 2):
                    if i + 1 < len(data):
                        header = data[i] if data[i].startswith('>') else f">{data[i]}"
                        sequence = data[i + 1]
                        f.write(f"{header}\n")
                        # Write sequence with line breaks every 80 characters
                        for j in range(0, len(sequence), 80):
                            f.write(f"{sequence[j:j+80]}\n")
            elif data is None:
                # For MUSCLE alignments, gget.muscle() returns None but prints to stdout
                # We need to capture the stdout or use a different approach
                f.write("# MUSCLE alignment completed\n# Output was printed to console\n")
            else:
                f.write(str(data))
    
    def _register_gget_tools(self):
        """Register selected gget tools only."""

        # Gene search and info
        self.tool(name=f"{self.prefix}search")(self.search_genes)
        self.tool(name=f"{self.prefix}info")(self.get_gene_info)

        # Sequence retrieval
        self.tool(name=f"{self.prefix}seq")(self.get_sequences)

        # BLAST sequence analysis
        self.tool(name=f"{self.prefix}blast")(self.blast_sequence)

        # Enrichr functional enrichment
        self.tool(name=f"{self.prefix}enrichr")(self.enrichr_analysis)

        # Single-cell expression via CELLxGENE
        self.tool(name=f"{self.prefix}cellxgene")(self.cellxgene_query)

    async def search_genes(
        self, 
        search_terms: Union[str, List[str]], 
        species: str = "homo_sapiens",
        release: Optional[int] = None,
        id_type: str = "gene",
        andor: str = "or",
        limit: Optional[int] = None
    ) -> SearchResult:
        """Search for genes using gene symbols, names, or synonyms.
        
        Use this tool FIRST when you have gene names/symbols and need to find their Ensembl IDs.
        Returns Ensembl IDs which are required for get_gene_info and get_sequences tools.
        
        Args:
            search_terms: Gene symbols, names, or synonyms as string or list of strings (e.g., 'TP53' or ['TP53', 'BRCA1'])
            species: Target species (e.g., 'homo_sapiens', 'mus_musculus') or specific core database name
            release: Ensembl release number (e.g., 104). Default: None (latest release)
            id_type: "gene" (default) or "transcript" - defines whether genes or transcripts are returned
            andor: "or" (default) or "and" - "or" returns genes with ANY searchword, "and" requires ALL searchwords
            limit: Maximum number of search results returned. Default: None (no limit)
        
        Returns:
            SearchResult: DataFrame with gene search results containing Ensembl IDs and descriptions
            
        Example:
            Input: search_terms='BRCA1', species='homo_sapiens'
            Output: DataFrame with columns like 'ensembl_id', 'gene_name', 'description'
        
        Downstream tools that need the Ensembl IDs from this search:
            - get_gene_info: Get detailed gene information  
            - get_sequences: Get DNA/protein sequences
        
        Note: Only searches in "gene name" and "description" sections of Ensembl database.
        """
        with start_action(action_type="gget_search", search_terms=search_terms, species=species):
            result = gget.search(
                searchwords=search_terms, 
                species=species, 
                release=release,
                id_type=id_type,
                andor=andor,
                limit=limit
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_gene_info(
        self, 
        ensembl_ids: Union[str, List[str]],
        ncbi: bool = True,
        uniprot: bool = True,
        pdb: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Get detailed gene and transcript metadata using Ensembl IDs.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs as string or list (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            ncbi: If True, includes data from NCBI. Default: True
            uniprot: If True, includes data from UniProt. Default: True  
            pdb: If True, also returns PDB IDs (might increase runtime). Default: False
            verbose: If True, prints progress information. Default: True
            
        Returns:
            Dict[str, Any]: DataFrame with gene information containing metadata from multiple databases
        
        Example workflow:
            1. search_genes('TP53', 'homo_sapiens') → get Ensembl ID 'ENSG00000141510'
            2. get_gene_info('ENSG00000141510') 
            
        Example output:
            DataFrame with columns like 'ensembl_id', 'symbol', 'biotype', 'chromosome', 'start', 'end', 
            plus NCBI, UniProt, and optionally PDB information
        """
        with start_action(action_type="gget_info", ensembl_ids=ensembl_ids):
            result = gget.info(
                ens_ids=ensembl_ids, 
                ncbi=ncbi,
                uniprot=uniprot,
                pdb=pdb,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_sequences(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False,
        isoforms: bool = False,
        verbose: bool = True
    ) -> SequenceResult:
        """Fetch nucleotide or amino acid sequence (FASTA) of genes or transcripts.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs as string or list (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            translate: If True, returns amino acid sequences; if False, returns nucleotide sequences. Default: False
                      Nucleotide sequences fetched from Ensembl REST API, amino acid from UniProt REST API
            isoforms: If True, returns sequences of all known transcripts (only for gene IDs). Default: False
            verbose: If True, prints progress information. Default: True
            
        Returns:
            SequenceResult: List containing the requested sequences in FASTA format
            Battle testing revealed the actual return is a list, not the various formats mentioned before
        
        Example workflow for protein sequence:
            1. search_genes('TP53', 'homo_sapiens') → 'ENSG00000141510'
            2. get_sequences('ENSG00000141510', translate=True)
            
        Example output:
            List of sequences in FASTA format: ['>ENSG00000141510', 'MEEPQSDPSVEPPLSQ...']
        
        Downstream tools that use protein sequences:
            - alphafold_predict: Predict 3D structure from protein sequence
            - blast_sequence: Search for similar sequences
        """
        with start_action(action_type="gget_seq", ensembl_ids=ensembl_ids, translate=translate):
            result = gget.seq(
                ens_ids=ensembl_ids, 
                translate=translate, 
                isoforms=isoforms,
                verbose=verbose
            )
            return result

    async def blast_sequence(
        self, 
        sequence: str,
        program: str = "default",
        database: str = "default",
        limit: int = 50,
        expect: float = 10.0,
        low_comp_filt: bool = False,
        megablast: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """BLAST a nucleotide or amino acid sequence against any BLAST database.
        
        Args:
            sequence: Nucleotide or amino acid sequence (string) or path to FASTA file
                     (If FASTA has multiple sequences, only first will be submitted)
            program: BLAST program - 'blastn', 'blastp', 'blastx', 'tblastn', or 'tblastx'
                    Default: "default" (auto-detects: 'blastn' for nucleotide, 'blastp' for amino acid)
            database: BLAST database - 'nt', 'nr', 'refseq_rna', 'refseq_protein', 'swissprot', 'pdbaa', 'pdbnt'
                     Default: "default" (auto-detects: 'nt' for nucleotide, 'nr' for amino acid)
            limit: Maximum number of hits to return. Default: 50
            expect: Expect value cutoff (float). Default: 10.0
            low_comp_filt: Apply low complexity filter. Default: False
            megablast: Use MegaBLAST algorithm (blastn only). Default: True
            verbose: Print progress information. Default: True
        
        Returns:
            Dict[str, Any]: DataFrame with BLAST results including alignment details and scores
            
        Example:
            Input: sequence="ATGCGATCGTAGC", program="blastn", database="nt"
            Output: DataFrame with BLAST hits, E-values, scores, and alignments
        
        Note: 
            - NCBI server rule: Run scripts weekends or 9pm-5am ET weekdays for >50 searches
            - More info on databases: https://ncbi.github.io/blast-cloud/blastdb/available-blastdbs.html
        """
        with start_action(action_type="gget_blast", sequence_length=len(sequence), program=program):
            result = gget.blast(
                sequence=sequence,
                program=program,
                database=database,
                limit=limit,
                expect=expect,
                low_comp_filt=low_comp_filt,
                megablast=megablast,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def enrichr_analysis(
        self, 
        genes: List[str],
        database: str = "KEGG_2021_Human",
        species: str = "human",
        background_list: Optional[List[str]] = None,
        background: bool = False,
        ensembl: bool = False,
        ensembl_bkg: bool = False,
        plot: bool = False,
        kegg_out: Optional[str] = None,
        kegg_rank: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform functional enrichment analysis on gene list using Enrichr.
        
        Args:
            genes: List of gene symbols (e.g., ['PHF14', 'RBM3']) or Ensembl IDs if ensembl=True
            database: Reference database or shortcuts for human/mouse:
                     'pathway' (KEGG_2021_Human), 'transcription' (ChEA_2016), 
                     'ontology' (GO_Biological_Process_2021), 'diseases_drugs' (GWAS_Catalog_2019),
                     'celltypes' (PanglaoDB_Augmented_2021), 'kinase_interactions' (KEA_2015)
                     Or full database name from https://maayanlab.cloud/Enrichr/#libraries
            species: Species database - 'human', 'mouse', 'fly', 'yeast', 'worm', 'fish'. Default: "human"
            background_list: Custom background genes (only for human/mouse). Default: None
            background: Use >20,000 default background genes (only for human/mouse). Default: False
            ensembl: If True, 'genes' are Ensembl gene IDs. Default: False
            ensembl_bkg: If True, 'background_list' are Ensembl gene IDs. Default: False
            plot: Create graphical overview of first 15 results. Default: False
            kegg_out: Path to save highlighted KEGG pathway image (e.g., 'path/kegg_pathway.png'). Default: None
            kegg_rank: Pathway rank to plot in KEGG image. Default: 1
            verbose: Print progress information. Default: True
        
        Returns:
            Dict[str, Any]: DataFrame with enrichment results including pathways, p-values, and statistical measures
            Battle testing confirmed functional analysis capabilities with cancer genes
            
        Example:
            Input: genes=['PHF14', 'RBM3', 'MSL1'], database='pathway'  
            Output: DataFrame with KEGG pathway enrichment results and statistics
        """
        with start_action(action_type="gget_enrichr", genes=genes, database=database):
            result = gget.enrichr(
                genes=genes,
                database=database,
                species=species,
                background_list=background_list,
                background=background,
                ensembl=ensembl,
                ensembl_bkg=ensembl_bkg,
                plot=plot,
                kegg_out=kegg_out,
                kegg_rank=kegg_rank,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def cellxgene_query(
        self, 
        species: str = "homo_sapiens",
        gene: Optional[Union[str, List[str]]] = None,
        ensembl: bool = False,
        column_names: List[str] = ["dataset_id", "assay", "suspension_type", "sex", "tissue_general", "tissue", "cell_type"],
        meta_only: bool = False,
        tissue: Optional[Union[str, List[str]]] = None,
        cell_type: Optional[Union[str, List[str]]] = None,
        development_stage: Optional[Union[str, List[str]]] = None,
        disease: Optional[Union[str, List[str]]] = None,
        sex: Optional[Union[str, List[str]]] = None,
        is_primary_data: bool = True,
        dataset_id: Optional[Union[str, List[str]]] = None,
        tissue_general_ontology_term_id: Optional[Union[str, List[str]]] = None,
        tissue_general: Optional[Union[str, List[str]]] = None,
        assay_ontology_term_id: Optional[Union[str, List[str]]] = None,
        assay: Optional[Union[str, List[str]]] = None,
        cell_type_ontology_term_id: Optional[Union[str, List[str]]] = None,
        development_stage_ontology_term_id: Optional[Union[str, List[str]]] = None,
        disease_ontology_term_id: Optional[Union[str, List[str]]] = None,
        donor_id: Optional[Union[str, List[str]]] = None,
        self_reported_ethnicity_ontology_term_id: Optional[Union[str, List[str]]] = None,
        self_reported_ethnicity: Optional[Union[str, List[str]]] = None,
        sex_ontology_term_id: Optional[Union[str, List[str]]] = None,
        suspension_type: Optional[Union[str, List[str]]] = None,
        tissue_ontology_term_id: Optional[Union[str, List[str]]] = None,
        census_version: str = "stable",
        verbose: bool = True,
        out: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query single-cell RNA-seq data from CZ CELLxGENE Discover using Census.
        
        NOTE: Querying large datasets requires >16 GB RAM and >5 Mbps internet connection.
        Use cell metadata attributes to define specific (sub)datasets of interest.
        
        Args:
            species: Target species - 'homo_sapiens' or 'mus_musculus'. Default: "homo_sapiens"
            gene: Gene name(s) or Ensembl ID(s) (e.g., ['ACE2', 'SLC5A1'] or ['ENSG00000130234']). Default: None
                  Set ensembl=True when providing Ensembl IDs
            ensembl: If True, genes are Ensembl IDs instead of gene names. Default: False
            column_names: Metadata columns to return in AnnData.obs. Default: ["dataset_id", "assay", "suspension_type", "sex", "tissue_general", "tissue", "cell_type"]
            meta_only: If True, returns only metadata DataFrame (AnnData.obs). Default: False
            tissue: Tissue(s) to query (e.g., ['lung', 'blood']). Default: None
            cell_type: Cell type(s) to query (e.g., ['mucus secreting cell']). Default: None
            development_stage: Development stage(s) to filter. Default: None
            disease: Disease(s) to filter. Default: None
            sex: Sex(es) to filter (e.g., 'female'). Default: None
            is_primary_data: If True, returns only canonical instance of cellular observation. Default: True
            dataset_id: CELLxGENE dataset ID(s) to query. Default: None
            tissue_general_ontology_term_id: High-level tissue UBERON ID(s). Default: None
            tissue_general: High-level tissue label(s). Default: None
            assay_ontology_term_id: Assay ontology term ID(s). Default: None
            assay: Assay type(s) as defined in CELLxGENE schema. Default: None
            cell_type_ontology_term_id: Cell type ontology term ID(s). Default: None
            development_stage_ontology_term_id: Development stage ontology term ID(s). Default: None
            disease_ontology_term_id: Disease ontology term ID(s). Default: None
            donor_id: Donor ID(s) as defined in CELLxGENE schema. Default: None
            self_reported_ethnicity_ontology_term_id: Ethnicity ontology ID(s). Default: None
            self_reported_ethnicity: Self-reported ethnicity. Default: None
            sex_ontology_term_id: Sex ontology ID(s). Default: None
            suspension_type: Suspension type(s) as defined in CELLxGENE schema. Default: None
            tissue_ontology_term_id: Tissue ontology term ID(s). Default: None
            census_version: Census version ('stable', 'latest', or specific date like '2023-05-15'). Default: "stable"
            verbose: Print progress information. Default: True
            out: Path to save AnnData h5ad file (or CSV when meta_only=True). Default: None
        
        Returns:
            Dict[str, Any]: AnnData object (when meta_only=False) or DataFrame (when meta_only=True)
                           with single-cell expression data and metadata
        
        Example:
            Input: gene=['ACE2'], tissue=['lung'], cell_type=['alveolar epithelial cell']
            Output: Single-cell expression data for ACE2 in lung alveolar epithelial cells
            
        Example (metadata only):
            Input: tissue=['brain'], meta_only=True
            Output: Metadata DataFrame for brain tissue datasets
        """
        with start_action(action_type="gget_cellxgene", genes=gene, tissues=tissue):
            result = gget.cellxgene(
                species=species,
                gene=gene,
                ensembl=ensembl,
                column_names=column_names,
                meta_only=meta_only,
                tissue=tissue,
                cell_type=cell_type,
                development_stage=development_stage,
                disease=disease,
                sex=sex,
                is_primary_data=is_primary_data,
                dataset_id=dataset_id,
                tissue_general_ontology_term_id=tissue_general_ontology_term_id,
                tissue_general=tissue_general,
                assay_ontology_term_id=assay_ontology_term_id,
                assay=assay,
                cell_type_ontology_term_id=cell_type_ontology_term_id,
                development_stage_ontology_term_id=development_stage_ontology_term_id,
                disease_ontology_term_id=disease_ontology_term_id,
                donor_id=donor_id,
                self_reported_ethnicity_ontology_term_id=self_reported_ethnicity_ontology_term_id,
                self_reported_ethnicity=self_reported_ethnicity,
                sex_ontology_term_id=sex_ontology_term_id,
                suspension_type=suspension_type,
                tissue_ontology_term_id=tissue_ontology_term_id,
                census_version=census_version,
                verbose=verbose,
                out=out
            )
            return result
