"""GEM reconstruction pipeline (doc/20 §6.5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from helixlang.plugins.annotation import GeneAnnotation
from helixlang.plugins.annotation.sequences import extract_protein_sequences
from helixlang.plugins.annotation.tf_detection import detect_transcription_factors
from helixlang.plugins.gem.biomass import build_biomass_reaction
from helixlang.plugins.gem.bottom_up import BottomUpResult, bottom_up_reconstruct
from helixlang.plugins.gem.consensus import ConsensusResult, consensus_merge
from helixlang.plugins.gem.gapfill import GapfillResult, gapfill
from helixlang.plugins.gem.grn_inference import GRNInferenceResult, infer_grn
from helixlang.plugins.gem.top_down import TopDownResult, top_down_reconstruct
from helixlang.plugins.kinetics.kcat_predictor import KcatPrediction, predict_kcat
from helixlang.plugins.kinetics.km_estimator import estimate_km


@dataclass
class GemPipelineResult:
    """Complete result of the six-stage GEM pipeline."""

    # Stage 2: Functional annotation
    annotations: dict[str, GeneAnnotation] = field(default_factory=dict)
    annotated_genes: int = 0

    # Stage 3: Metabolic network
    bottom_up: BottomUpResult | None = None
    top_down: TopDownResult | None = None
    consensus: ConsensusResult | None = None
    gapfill: GapfillResult | None = None

    # Stage 4: GRN
    tf_result: object | None = None  # TFScanResult
    grn: GRNInferenceResult | None = None

    # Stage 5: Kinetics
    kcat_predictions: list[KcatPrediction] = field(default_factory=list)
    km_estimates: dict[str, float] = field(default_factory=dict)

    # Stage 6: Integration & FBA
    metabolic_model: Any = None  # metabolism.MetabolicModel
    growth_rate: float = 0.0
    fba_fluxes: dict[str, float] = field(default_factory=dict)
    fba_analysis: dict[str, Any] = field(default_factory=dict)
    biomass_reaction: object | None = None
    final_reaction_count: int = 0
    final_gene_count: int = 0

    # Status
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stages_completed: int = 0

    def summary(self) -> str:
        lines = [
            "GEM Pipeline Summary",
            f"{'=' * 40}",
            f"Genes annotated:    {self.annotated_genes}",
            f"Reactions (bottom): {self.bottom_up.reaction_count if self.bottom_up else 0}",
            f"Reactions (top):    {self.top_down.reaction_count if self.top_down else 0}",
            f"Reactions (final):  {self.final_reaction_count}",
            f"Genes in model:     {self.final_gene_count}",
            f"GRN edges:          {self.grn.total_edges if self.grn else 0}",
            f"k_cat predictions:  {len(self.kcat_predictions)}",
            f"Km estimates:       {len(self.km_estimates)}",
            f"Growth rate (h⁻¹): {self.growth_rate:.4f}",
            f"Stages completed:   {self.stages_completed}/6",
        ]
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


def _annotate_from_fasta(
    fasta_path: str,
    gff3_path: str | None = None,
    diamond_db: str | None = None,
) -> dict[str, GeneAnnotation]:
    """Annotate genes from genome using DIAMOND blastp + web API fallback.

    Strategy:
    1. Extract protein sequences from genome (GFF3 → translate, or direct FASTA)
    2. Run DIAMOND blastp against UniRef90/SwissProt if database available
    3. Parse EC numbers from DIAMOND hits (UniProt DR lines or header parsing)
    4. Fall back to UniProt REST API if no local database
    5. Fall back to header-based heuristic if all else fails

    Parameters
    ----------
    fasta_path : path to genome FASTA (nucleotide or protein)
    gff3_path : optional path to GFF3 annotation
    diamond_db : optional path to DIAMOND database (.dmnd)

    Returns
    -------
    dict mapping gene_id → GeneAnnotation
    """
    annotations: dict[str, GeneAnnotation] = {}

    # Step 1: Extract protein sequences
    proteins = extract_protein_sequences(fasta_path, gff3_path)
    if not proteins:
        # Try header-based extraction as fallback
        try:
            with open(fasta_path) as fh:
                for line in fh:
                    if not line.startswith(">"):
                        continue
                    header = line[1:].strip()
                    gene_id = header.split()[0] if header else ""
                    if gene_id:
                        proteins[gene_id] = ""
        except FileNotFoundError:  # SILENTBENIGN - optional protein FASTA
            pass

    # Step 2: Try DIAMOND blastp if database available
    if diamond_db and proteins:
        try:
            # Write proteins to temp FASTA for DIAMOND
            import tempfile
            from pathlib import Path

            from helixlang.plugins.annotation.blast import run_diamond

            tmp_fasta = Path(tempfile.mktemp(suffix=".fasta"))
            with open(tmp_fasta, "w") as fh:
                for gid, seq in proteins.items():
                    fh.write(f">{gid}\n{seq}\n")

            diamond_result = run_diamond(tmp_fasta, diamond_db)
            tmp_fasta.unlink(missing_ok=True)

            # Parse EC numbers from DIAMOND hits
            for gene_id in proteins:
                hits = diamond_result.hits_for(gene_id)
                ec_numbers = []
                kegg_ko = []

                for hit in hits:
                    # Parse EC number from subject title
                    # UniProt format: "tr|A0A0...|A0A0... EC:1.2.3.4 ..."
                    ec = _extract_ec_from_hit(hit.stitle)
                    if ec:
                        ec_numbers.append(ec)

                    # Parse KO term from subject ID
                    if hit.subject_id.startswith("K"):
                        kegg_ko.append(hit.subject_id)

                annotations[gene_id] = GeneAnnotation(
                    gene_id=gene_id,
                    protein_seq=proteins.get(gene_id, ""),
                    ec_numbers=ec_numbers,
                    kegg_ko=kegg_ko,
                    go_terms=[],
                    confidence=0.8 if ec_numbers else 0.3,
                )

            return annotations

        except FileNotFoundError:  # SILENTBENIGN - DIAMOND not installed
            pass  # DIAMOND not installed
        except Exception:  # SILENTBENIGN - fall through to UniProt
            pass  # DIAMOND failed, continue to fallback

    # Step 3: Try UniProt REST API fallback (lightweight mode)
    if proteins:
        annotations = _annotate_via_uniprot_api(proteins)
        if annotations:
            return annotations

    # Step 4: Header-based heuristic (last resort)
    for gene_id, seq in proteins.items():
        annotations[gene_id] = GeneAnnotation(
            gene_id=gene_id,
            protein_seq=seq,
            ec_numbers=[],
            kegg_ko=[],
            go_terms=[],
            confidence=0.1,
        )

    return annotations


def _extract_ec_from_hit(title: str) -> str | None:
    """Extract EC number from DIAMOND hit title.

    Handles formats:
    - "tr|A0A0...|A0A0... EC:1.2.3.4 ..."
    - "sp|P00000|AAAA E=1.2.3.4"
    - "1.2.3.4" (bare EC number)
    """
    import re

    # Look for EC:x.x.x.x pattern
    match = re.search(r"EC[:\s]+(\d+\.\d+\.\d+\.\d+)", title)
    if match:
        return match.group(1)

    # Look for bare EC number at end
    match = re.search(r"(\d+\.\d+\.\d+\.\d+)\s*$", title)
    if match:
        return match.group(1)

    return None


def _annotate_via_uniprot_api(
    proteins: dict[str, str],
) -> dict[str, GeneAnnotation]:
    """Annotate genes via UniProt REST API (lightweight fallback).

    Uses UniProt's ID Mapping API first (fast, for UniProt accessions),
    then falls back to NCBI BLAST (for real genome gene IDs like WP_xxx).
    Finally, uses sequence-based homology search if both fail.

    Parameters
    ----------
    proteins : dict mapping gene_id → protein sequence

    Returns
    -------
    dict mapping gene_id → GeneAnnotation (empty if API unavailable)
    """
    annotations: dict[str, GeneAnnotation] = {}

    # Strategy 1: Try ID Mapping (fast, works for UniProt accessions)
    annotations = _annotate_via_uniprot_idmapping(proteins)
    if annotations and len(annotations) > len(proteins) * 0.3:
        return annotations

    # Strategy 2: Try NCBI BLAST (works for real genome gene IDs)
    unannotated = {
        gid: seq for gid, seq in proteins.items()
        if gid not in annotations and seq
    }
    if unannotated:
        blast_annotations = _annotate_via_ncbi_blast(unannotated)
        annotations.update(blast_annotations)

    # Strategy 3: Sequence similarity via UniProt REST (for remaining)
    still_unannotated = {
        gid: seq for gid, seq in proteins.items()
        if gid not in annotations and seq
    }
    if still_unannotated:
        seq_annotations = _annotate_via_uniprot_sequence(still_unannotated)
        annotations.update(seq_annotations)

    return annotations


def _annotate_via_uniprot_idmapping(
    proteins: dict[str, str],
) -> dict[str, GeneAnnotation]:
    """Strategy 1: UniProt ID Mapping API (fast, for UniProt accessions)."""
    import json
    import time
    import urllib.request

    annotations: dict[str, GeneAnnotation] = {}
    gene_ids = list(proteins.keys())[:100]

    try:
        query_data = json.dumps({
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": gene_ids,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://rest.uniprot.org/idmapping/run",
            data=query_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            job = json.loads(resp.read())
            job_id = job.get("jobId", "")
            if not job_id:
                return annotations

        time.sleep(1)
        for _ in range(10):
            req = urllib.request.Request(
                f"https://rest.uniprot.org/idmapping/status/{job_id}"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = json.loads(resp.read())
                if status.get("jobStatus") == "FINISHED":
                    break
                time.sleep(1)

        req = urllib.request.Request(
            f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read())

        for result in results.get("results", []):
            from_id = result.get("from", "")
            to_data = result.get("to", {})
            if isinstance(to_data, dict):
                accessions = to_data.get("primaryAccession", "")
                if not accessions:
                    acc_list = to_data.get("accession", [])
                    accessions = acc_list[0] if isinstance(acc_list, list) and acc_list else ""
                if not accessions:
                    continue

                try:
                    req = urllib.request.Request(
                        f"https://rest.uniprot.org/uniprotkb/{accessions}.json"
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        entry = json.loads(resp.read())

                    ec_numbers = []
                    for ref in entry.get("proteinDescription", {}).get(
                        "recommendedName", {}
                    ).get("ecNumbers", []):
                        ec = ref.get("value", "")
                        if ec:
                            ec_numbers.append(ec)

                    go_terms = []
                    for xref in entry.get("uniProtKBCrossReferences", []):
                        if xref.get("database") == "GO":
                            go_terms.append(xref.get("id", ""))

                    annotations[from_id] = GeneAnnotation(
                        gene_id=from_id,
                        protein_seq=proteins.get(from_id, ""),
                        ec_numbers=ec_numbers,
                        kegg_ko=[],
                        go_terms=go_terms,
                        confidence=0.7 if ec_numbers else 0.3,
                    )
                except Exception:
                    continue

        time.sleep(0.5)

    except Exception:  # SILENTBENIGN - best-effort annotation download
        pass

    return annotations


def _annotate_via_ncbi_blast(
    proteins: dict[str, str],
) -> dict[str, GeneAnnotation]:
    """Strategy 2: NCBI BLASTP against nr database.

    Works for real genome gene IDs (RefSeq WP_xxx, etc.) by doing
    sequence similarity search against NCBI's non-redundant database.
    """
    import json
    import tempfile
    import time
    import urllib.parse
    import urllib.request
    from pathlib import Path

    annotations: dict[str, GeneAnnotation] = {}

    # Write proteins to temp FASTA
    tmp_fasta = Path(tempfile.mktemp(suffix=".fasta"))
    try:
        with open(tmp_fasta, "w") as fh:
            for gid, seq in list(proteins.items())[:20]:
                fh.write(f">{gid}\n{seq}\n")

        # Submit BLAST job to NCBI
        fasta_content = tmp_fasta.read_text()
        params = urllib.parse.urlencode({
            "CMD": "Put",
            "DATABASE": "nr",
            "PROGRAM": "blastp",
            "QUERY": fasta_content,
            "FORMAT_TYPE": "JSON2",
            "EXPECT": "0.001",
            "HITLIST_SIZE": "5",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi",
            data=params,
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract RID
        rid = ""
        for line in html.split("\n"):
            if "RID =" in line:
                rid = line.split("RID =")[1].split()[0].strip()
                break
        if not rid:
            return annotations

        # Poll for completion
        time.sleep(5)
        for _ in range(30):
            req = urllib.request.Request(
                f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=JSON2&RID={rid}"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result_text = resp.read().decode("utf-8", errors="ignore")
            if "Status=ready" in result_text or "\"blastoutput\"" in result_text.lower():
                break
            time.sleep(3)

        # Parse JSON results
        try:
            result_data = json.loads(result_text)
            for query_result in result_data.get("blastoutput", {}).get(
                "results", {}
            ).get("blasts", []):
                query_id = query_result.get("query_title", "").split()[0]
                ec_numbers = []
                for hit in query_result.get("hits", [])[:3]:
                    title = hit.get("title", "")
                    hit_ec = _extract_ec_from_blast_title(title)
                    if hit_ec:
                        ec_numbers.append(hit_ec)
                        break

                if query_id in proteins:
                    annotations[query_id] = GeneAnnotation(
                        gene_id=query_id,
                        protein_seq=proteins[query_id],
                        ec_numbers=ec_numbers,
                        kegg_ko=[],
                        go_terms=[],
                        confidence=0.6 if ec_numbers else 0.2,
                    )
        except (json.JSONDecodeError, KeyError):  # SILENTBENIGN - malformed reply
            pass

    except Exception:  # SILENTBENIGN - best-effort BLAST annotation
        pass
    finally:
        tmp_fasta.unlink(missing_ok=True)

    return annotations


def _extract_ec_from_blast_title(title: str) -> str | None:
    """Extract EC number from BLAST hit title."""
    import re

    # UniProt format: "recName: Full=... EC=x.x.x.x"
    match = re.search(r"EC[:\s]+(\d+\.\d+\.\d+\.\d+)", title)
    if match:
        return match.group(1)

    # KEGG format: "EC:x.x.x.x"
    match = re.search(r"\bEC:(\d+\.\d+\.\d+\.\d+)\b", title)
    if match:
        return match.group(1)

    # Bare EC number
    match = re.search(r"\b(\d+\.\d+\.\d+\.\d+)\b", title)
    if match:
        return match.group(1)

    return None


def _annotate_via_uniprot_sequence(
    proteins: dict[str, str],
) -> dict[str, GeneAnnotation]:
    """Strategy 3: UniProt sequence search (slow, last resort).

    Uses UniProt's REST API to search by protein sequence similarity.
    Limited to 5 sequences per batch to avoid timeout.
    """
    import json
    import time
    import urllib.request

    annotations: dict[str, GeneAnnotation] = {}

    # Process in small batches
    items = list(proteins.items())[:10]
    for gene_id, sequence in items:
        if not sequence or len(sequence) < 10:
            continue

        try:
            # Use UniProt search with sequence
            query = f"sequence:{sequence[:50]}"
            url = (
                f"https://rest.uniprot.org/uniprotkb/search"
                f"?query={urllib.parse.quote(query)}"
                f"&format=json&size=1"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            results = data.get("results", [])
            if results:
                entry = results[0]
                ec_numbers = []
                for ref in entry.get("proteinDescription", {}).get(
                    "recommendedName", {}
                ).get("ecNumbers", []):
                    ec = ref.get("value", "")
                    if ec:
                        ec_numbers.append(ec)

                annotations[gene_id] = GeneAnnotation(
                    gene_id=gene_id,
                    protein_seq=sequence,
                    ec_numbers=ec_numbers,
                    kegg_ko=[],
                    go_terms=[],
                    confidence=0.5 if ec_numbers else 0.1,
                )

            time.sleep(0.3)

        except Exception:
            continue

    return annotations


def _infer_substrate_from_ec(ec_number: str) -> str:
    """Infer primary substrate from EC number class.

    Parameters
    ----------
    ec_number : EC number (e.g., "1.1.1.1")

    Returns
    -------
    Substrate name or empty string if unknown
    """
    # Map EC class 1 (oxidoreductases) to common substrates
    ec_prefix = ec_number[:3] if len(ec_number) >= 3 else ec_number

    EC_SUBSTRATE_MAP = {
        "1.1": "NAD",
        "1.2": "NAD",
        "1.3": "NAD",
        "1.8": "NAD",
        "1.9": "O2",
        "2.1": "ATP",
        "2.2": "ATP",
        "2.3": "CoA",
        "2.4": "ATP",
        "2.5": "ATP",
        "2.6": "Glutamate",
        "2.7": "ATP",
        "3.1": "H2O",
        "3.2": "H2O",
        "3.5": "H2O",
        "4.1": "H2O",
        "4.2": "H2O",
        "5.1": "H2O",
        "5.3": "H2O",
        "5.4": "H2O",
        "6.1": "ATP",
        "6.2": "ATP",
        "6.4": "ATP",
        "7.1": "H2O",
        "7.2": "ADP",
    }

    return EC_SUBSTRATE_MAP.get(ec_prefix, "")


def run_gem_pipeline(
    genome_fasta: str,
    organism: str = "e_coli_k12",
    use_database_interactions: bool = True,
    include_spontaneous: bool = True,
    run_gapfill: bool = True,
    target_organism: str = "Escherichia coli",
    gff3_path: str | None = None,
    diamond_db: str | None = None,
    medium: str = "glucose_minimal",
) -> GemPipelineResult:
    """Execute the complete six-stage GEM reconstruction pipeline.

    Stage 1: Genome import & preprocessing (input: FASTA)
    Stage 2: Functional annotation (annotation/ modules)
    Stage 3: Metabolic network reconstruction (gem/ modules)
    Stage 4: GRN inference (gem/grn_inference.py)
    Stage 5: Kinetic parameter estimation (kinetics/ modules)
    Stage 6: Integration and validation

    Parameters
    ----------
    genome_fasta : path to genome FASTA file
    organism : organism identifier
    use_database_interactions : use known regulatory interactions
    include_spontaneous : include spontaneous reactions
    run_gapfill : run gap-filling procedure
    target_organism : target organism for BRENDA lookup
    gff3_path : optional path to GFF3 annotation file
    diamond_db : optional path to DIAMOND database (.dmnd)

    Returns
    -------
    GemPipelineResult
    """
    result = GemPipelineResult()

    # Stage 1: Genome import (validate FASTA exists)
    try:
        with open(genome_fasta) as fh:
            first_line = fh.readline()
            if not first_line.startswith(">"):
                result.errors.append(
                    f"Invalid FASTA file: {genome_fasta}"
                )
                return result
    except FileNotFoundError:
        result.errors.append(f"Genome file not found: {genome_fasta}")
        return result

    # Stage 2: Functional annotation
    try:
        annotations = _annotate_from_fasta(
            genome_fasta,
            gff3_path=gff3_path,
            diamond_db=diamond_db,
        )
        result.annotations = annotations
        result.annotated_genes = len(annotations)
        result.stages_completed = 2
    except Exception as exc:
        result.errors.append(f"Stage 2 failed: {exc}")
        return result

    # Stage 3: Metabolic network reconstruction
    try:
        result.bottom_up = bottom_up_reconstruct(
            annotations, include_spontaneous=include_spontaneous
        )
        result.top_down = top_down_reconstruct(annotations)
        result.consensus = consensus_merge(
            result.bottom_up, result.top_down
        )
        if run_gapfill:
            result.gapfill = gapfill(result.consensus)
        result.final_reaction_count = (
            result.consensus.reaction_count
            + (result.gapfill.gap_filled_count if result.gapfill else 0)
        )
        result.final_gene_count = result.consensus.from_bottom_up_only + result.consensus.from_both
        result.stages_completed = 3
    except Exception as exc:
        result.errors.append(f"Stage 3 failed: {exc}")
        return result

    # Stage 4: GRN inference
    try:
        tf_result = detect_transcription_factors(genome_fasta)
        result.tf_result = tf_result
        # Pass genome gene IDs for validation: skip database edges
        # whose TF or target is not in the input genome.
        _genome_gene_ids: set[str] | None = None
        if result.annotations:
            _genome_gene_ids = set(result.annotations.keys())
        # Convert pipeline bool to database_interactions list or None
        # None = use default KNOWN_REGULATORY_INTERACTIONS
        # [] = disable database edges entirely
        _db_interactions: list | None = (
            None if use_database_interactions else []
        )
        result.grn = infer_grn(
            tf_result,
            genome_fasta,
            database_interactions=_db_interactions,
            use_motif_prediction=True,
            genome_gene_ids=_genome_gene_ids,
        )
        result.stages_completed = 4
    except Exception as exc:
        result.warnings.append(f"Stage 4 failed (non-fatal): {exc}")

    # Stage 5: Kinetic parameter estimation
    # Build reverse mapping: reaction_id → gene_id (from GPR rules)
    rxn_to_gene: dict[str, str] = {}
    if result.consensus:
        for gene_id, ann in result.annotations.items():
            if ann.ec_numbers:
                # Map EC numbers to reactions via ec_mapping
                from helixlang.plugins.annotation.ec_mapping import ECOLI_CORE_EC_REACTIONS
                for ec in ann.ec_numbers:
                    for rxn_id in ECOLI_CORE_EC_REACTIONS.get(ec, []):
                        if rxn_id not in rxn_to_gene:
                            rxn_to_gene[rxn_id] = gene_id

    try:
        kcat_predictions = []
        km_estimates = {}
        for rxn_id in (result.consensus.reaction_ids()
                       if result.consensus else []):
            # Get EC number and sequence for this reaction
            gene_id = rxn_to_gene.get(rxn_id, "")
            gene_ann: GeneAnnotation | None = result.annotations.get(gene_id)
            ec_number = gene_ann.ec_numbers[0] if gene_ann and gene_ann.ec_numbers else ""
            sequence = gene_ann.protein_seq if gene_ann else ""

            kcat_pred = predict_kcat(
                rxn_id,
                ec_number=ec_number,
                sequence=sequence,
                target_organism=target_organism,
            )
            kcat_predictions.append(kcat_pred)

            # Get substrate for Km estimation
            substrate = ""
            if gene_ann and gene_ann.ec_numbers:
                # Try to infer substrate from EC class
                substrate = _infer_substrate_from_ec(gene_ann.ec_numbers[0])

            km_estimates[rxn_id] = estimate_km(
                rxn_id,
                substrate=substrate,
                sequence=sequence,
                target_organism=target_organism,
            )
        result.kcat_predictions = kcat_predictions
        result.km_estimates = km_estimates
        result.stages_completed = 5
    except Exception as exc:
        result.warnings.append(f"Stage 5 failed (non-fatal): {exc}")

    # Stage 6: Integration — build functional MetabolicModel, run FBA
    try:
        from helixlang.plugins.gem.bridge import (
            build_enzyme_capacity,
            build_functional_model,
        )

        result.biomass_reaction = build_biomass_reaction(organism)

        if result.consensus and result.consensus.reactions:
            # Build a functional model with core metabolism, transport,
            # and filtered biomass (Phase F of doc/22)
            model = build_functional_model(
                consensus=result.consensus,
                gapfill=result.gapfill,
                organism=organism,
                medium=medium,
            )
            result.metabolic_model = model

            # Read growth rate and fluxes from the model
            result.growth_rate = getattr(model, "_growth_rate", 0.0)
            result.fba_fluxes = getattr(model, "_fba_fluxes", {})

            # Wire enzyme capacity if we have kcat predictions
            if result.kcat_predictions and result.consensus:
                from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis
                fba = FluxBalanceAnalysis(model)
                enzyme_cap = build_enzyme_capacity(
                    result.consensus,
                    result.kcat_predictions,
                    enzyme_scale=1.0,
                )
                fba.set_enzyme_capacity(enzyme_cap)
                # Re-solve with enzyme capacity
                try:
                    fluxes = fba.solve(
                        objective=model.biomass_reaction)
                    result.fba_fluxes = fluxes
                    result.growth_rate = max(
                        0.0, fluxes.get(
                            model.biomass_reaction or "", 0.0))
                except Exception:  # SILENTBENIGN - keep non-enzyme result
                    pass  # keep the non-enzyme-capacity result

        result.stages_completed = 6
    except Exception as exc:
        result.warnings.append(f"Stage 6 failed (non-fatal): {exc}")

    return result
