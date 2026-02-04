# PubChem Field Survey (HTTP / PUG REST) - 2026-01-18

Goal: empirically verify which PubChem fields are available via HTTP so we can define the supported
scope for an in-agent PubChem tool (numeric facts + optional experimental properties).

This survey uses the same integration style as our current code (`src/tools/pubchem.py`): direct HTTP
requests to PubChem PUG REST / PUG-View endpoints (no third-party Python package).

## Endpoints Used

CID resolution (name -> CID):

- `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/<NAME>/cids/JSON`

Property table (CID -> descriptor fields):

- `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/<CID>/property/<PROP_LIST>/JSON`

PUG-View (CID -> experimental property sections, pKa/solubility/melting point, etc.):

- `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/<CID>/JSON?heading=<HEADING>`

## Key Observations (Important for Tool Design)

- Even when requesting `CanonicalSMILES`/`IsomericSMILES`, the property table often returns keys
  named `SMILES` and `ConnectivitySMILES` (not always `CanonicalSMILES` as a key).
- Some fields can be missing depending on compound form (e.g., ionic salts):
  - Example: `XLogP` was absent for sodium acetate (CID 517045) even when requested.
- Many "experimental" properties (e.g., melting point) are NOT available via the property-table API:
  requesting `BoilingPoint` / `MeltingPoint` via `/property/...` returns HTTP 400 `Invalid property`.
  These need PUG-View headings.

## Property Table Survey Results

Request template used (representative):

`/property/CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight,ExactMass,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,Charge,Complexity/JSON`

Notes:
- `MolecularWeight` and `ExactMass` often come back as strings.
- `XLogP`, `TPSA`, `Complexity`, counts are typically numeric.

| Query | CID | Formula | MolecularWeight | XLogP | TPSA | HBD | HBA | RotB | Charge | Complexity | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4-aminobenzoic acid | 978 | C7H7NO2 | 137.14 | 0.8 | 63.3 | 2 | 3 | 1 | 0 | 128 | SMILES key returned as `SMILES` + `ConnectivitySMILES` |
| benzoic acid | 243 | C7H6O2 | 122.12 | 1.9 | 37.3 | 1 | 2 | 1 | 0 | 103 | - |
| ethanol | 702 | C2H6O | 46.07 | -0.1 | 20.2 | 1 | 1 | 0 | 0 | 2 | negative control (no -COOH) |
| acetic acid | 176 | C2H4O2 | 60.05 | -0.2 | 37.3 | 1 | 2 | 0 | 0 | 31 | - |
| terephthalic acid | 7489 | C8H6O4 | 166.13 | 2.0 | 74.6 | 2 | 4 | 2 | 0 | 169 | - |
| citric acid | 311 | C6H8O7 | 192.12 | -1.7 | 132.0 | 4 | 7 | 5 | 0 | 227 | - |
| maleic acid | 444266 | C4H4O4 | 116.07 | -0.3 | 74.6 | 2 | 4 | 2 | 0 | 119 | `SMILES` includes stereo (`\\`), `ConnectivitySMILES` drops stereo |
| fumaric acid | 444972 | C4H4O4 | 116.07 | -0.3 | 74.6 | 2 | 4 | 2 | 0 | 119 | stereo differs from maleic (SMILES) |
| oxalic acid | 971 | C2H2O4 | 90.03 | -0.3 | 74.6 | 2 | 4 | 1 | 0 | 71 | - |
| succinic acid | 1110 | C4H6O4 | 118.09 | -0.6 | 74.6 | 2 | 4 | 3 | 0 | 92 | - |
| trifluoroacetic acid | 6422 | C2HF3O2 | 114.02 | 0.9 | 37.3 | 1 | 5 | 0 | 0 | 83 | HBA=5 reported by PubChem property table |
| sodium acetate | 517045 | C2H3NaO2 | 82.03 | (missing) | 40.1 | 0 | 2 | 0 | 0 | 34 | `XLogP` missing in response |
| 4-nitrobenzoic acid | 6108 | C7H5NO4 | 167.12 | 1.9 | 83.1 | 1 | 4 | 1 | 0 | 190 | - |
| glycine | 750 | C2H5NO2 | 75.07 | -3.2 | 63.3 | 2 | 3 | 1 | 0 | 42 | zwitterion in reality; property table returns Charge=0 |

## PUG-View Survey (Experimental Properties)

These headings DO return numeric values (as text), but require extraction/parsing.

### Dissociation Constants (pKa) Examples

Benzoic acid (CID 243), heading `Dissociation Constants`:
- pKa examples found in JSON: `4.19 (at 25 degC)`, `pKa = 4.204 at 25 degC` (multiple sources appear)

4-aminobenzoic acid (CID 978), heading `Dissociation Constants`:
- pKa examples found in JSON: `pKa1 = 2.38 at 25 degC`, `pKa2 = 4.85 at 25 degC`

### Solubility Examples

Benzoic acid (CID 243), heading `Solubility`:
- `0.29 g/L of benzoic acid in water at 20 degC`
- `In water, 3.5X10+3 mg/L at 25 degC` (i.e., 3.5 g/L)
- `less than 1 mg/mL at 68 degF` (source-specific; mixed units)

4-aminobenzoic acid (CID 978), heading `Solubility`:
- `In water, 5,390 mg/L at 25 degC; 6,110 mg/L at 30 degC`
- `6.11 mg/mL`

### Melting Point Examples

Benzoic acid (CID 243), heading `Melting Point`:
- `122.35 degC`, `122.4 degC`, and a range `121,5 - 123,5 degC` (locale-dependent formatting)

4-aminobenzoic acid (CID 978), heading `Melting Point`:
- `188.5 degC`

## Proposed Supported Scope for an Agent PubChem Tool (Based on This Survey)

### Tier 1: Reliable numeric descriptors via Property Table

Confirmed working on multiple compounds above:
- `MolecularFormula`
- `MolecularWeight`
- `ExactMass`
- `XLogP` (may be missing for ionic salts)
- `TPSA`
- `HBondDonorCount`
- `HBondAcceptorCount`
- `RotatableBondCount`
- `HeavyAtomCount`
- `Charge`
- `Complexity`
- Structure identifiers: `InChIKey`, plus `SMILES` / `ConnectivitySMILES` (returned keys)
- Optional (confirmed separately): `InChI`, `IUPACName`

### Tier 2: Experimental properties via PUG-View headings (requires parsing)

Confirmed headings with numeric values:
- `Dissociation Constants` (pKa values)
- `Solubility`
- `Melting Point`

### Out of scope for Property Table (invalid property)

Confirmed invalid when requested via `/property/...`:
- `BoilingPoint`
- `MeltingPoint`

These require PUG-View (or other sources).
