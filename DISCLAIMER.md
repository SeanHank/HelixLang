# Disclaimer

## No Warranty

HelixLang is provided "as is" and "as available" without warranty of any kind,
whether express, implied, or statutory. The authors and contributors expressly
disclaim all warranties, including but not limited to the implied warranties of
merchantability, fitness for a particular purpose, and noninfringement.

## Not Medical or Clinical Advice

HelixLang is a **research and educational software tool** for biological
simulation and domain-specific language experimentation. It is **not** a medical
device, clinical decision support system, or diagnostic tool. Nothing in
HelixLang constitutes medical, clinical, or pharmaceutical advice.

**Do not** use HelixLang output for:
- Diagnosing or treating any disease or medical condition.
- Making clinical decisions about patient care.
- Determining drug dosages, treatment protocols, or therapeutic strategies.
- Any purpose that requires regulatory approval (e.g., FDA, EMA, NMPA).

Always consult qualified healthcare professionals for medical decisions.

## Not a Substitute for Peer-Reviewed Research

HelixLang includes simulation backends that model biological systems (gene
regulation, metabolism, pharmacokinetics, disease dynamics). These models are
**simplified abstractions** of biological reality, not faithful reproductions of
living systems. Parameter values and citations are provided for transparency,
but:

- Model outputs should not be treated as experimentally validated results.
- Quantitative predictions (IC50, EC50, pharmacokinetic parameters, growth
  rates) are approximate and may deviate substantially from in vivo or in vitro
  measurements.
- Biological systems exhibit stochasticity, inter-individual
  variability, and emergent behaviors that may not be captured by any
  computational model.

Always validate simulation results against independent experimental data before
drawing scientific conclusions.

## Limitation of Liability

In no event shall the authors, contributors, or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the
use or other dealings in the software.

## Third-Party Dependencies

HelixLang may depend on third-party libraries (numpy, scipy, Flask, COBRApy,
PyTorch, RDKit, etc.) that are subject to their own licenses and warranties.
HelixLang does not warrant the availability, accuracy, or fitness for purpose
of any third-party dependency.

## Scientific Validation

HelixLang includes a scientific validation suite (`validation/`) that compares
simulation outputs against published literature and analytical solutions. These
validations are provided for **transparency and reproducibility**, not as
guarantees of accuracy:

- **All** benchmarks pass (all with full optional dependencies).
- **Median quantitative error** is approximately 3.0% against published values,
  but individual benchmarks may have higher errors.
- Validation results depend on the specific versions of dependencies used and
  the hardware/software environment.

Users should run the validation suite in their own environment and interpret
results in the context of their specific use case.

## Biologically-Inspired Computation

HelixLang uses biological metaphors (DNA, codons, genes, cells, regulatory
networks) as computational abstractions. These are **not** intended to model or
represent actual biological organisms unless explicitly stated in the
documentation for specific simulation backends (e.g., the GEM or PK/PD
modules). The biological inspiration is a design paradigm, not a claim of
biological fidelity.

## Regulatory Compliance

HelixLang has not been reviewed or approved by any regulatory agency. It is not
intended for use in any context that requires regulatory compliance, including
but not limited to:

- Pharmaceutical development or drug approval processes.
- Clinical trial design or analysis.
- Medical device software (SaMD).
- Diagnostic or prognostic tools.

Users are solely responsible for ensuring their use of HelixLang complies with
all applicable laws and regulations.

## Changes to This Disclaimer

This disclaimer may be updated from time to time. The most current version is
available in the project repository. Continued use of HelixLang after changes
constitutes acceptance of the updated disclaimer.

---

**Last updated:** August 2026
**Project:** [https://github.com/SeanHank/HelixLang](https://github.com/SeanHank/HelixLang)
**License:** GNU Affero General Public License v3.0 (see [LICENSE](LICENSE))
