# Data Provenance and Licenses

## Purpose

This document records the origin, access conditions, citation requirements, and redistribution status of the datasets and analytical materials included in the CIRN supplementary and reproducibility archive.

This is a working record for dissertation-committee and institutional review. Items marked **To verify** must be resolved before the archive is made public.

## General rules

1. Simulated datasets created specifically for the CIRN study may be archived with their generation procedures, parameter values, and software information.
2. Third-party datasets must retain their original source citation, accession or persistent identifier when available, retrieval date, and license or terms of use.
3. Raw third-party data should not be redistributed when permission is absent, unclear, or restricted.
4. Derived or processed data may be shared only when the original terms permit redistribution of derivative products.
5. No confidential, personally identifying, credential-bearing, or access-restricted information should be included in the public release.

## Component-level record

### 01 BIB Supplementary Information
- Material type: Author-generated supplementary documentation
- Redistribution status: Subject to final coauthor and institutional approval

### 02 Classical Lotka–Volterra
- Source: Author-generated deterministic simulation
- Redistribution status: Expected to be redistributable
- Preserve: equations, parameters, initial conditions, solver, time interval, and code version

### 03 Predator–Prey With Lags
- Source: Derived from the author-generated classical Lotka–Volterra simulation
- Redistribution status: Expected to be redistributable
- Preserve: tested lag values, common base dataset, settings, and reported figure correspondence

### 04 Stochastic Predator–Prey
- Source: Author-generated stochastic simulation
- Redistribution status: Expected to be redistributable
- Preserve: stochastic equations, integration scheme, parameters, initial conditions, random seed, and software version

### 05 Basic Network
- Source: Author-generated representative dynamical-system simulations
- Redistribution status: Expected to be redistributable
- Preserve: system-to-equation and system-to-figure mapping

### 06 Repressilator
- Source: Synthetic repressilator model or dataset used in the CIRN study
- Redistribution status: To verify
- Required action: Record whether the trajectory was author-generated or obtained from a third-party source; include the original model citation and applicable reuse terms

### 07 GeneNetWeaver E. coli Network
- Source: GeneNetWeaver-generated benchmark
- Redistribution status: To verify
- Required action: Cite GeneNetWeaver and the source network; record generation settings, stochasticity level, date, and redistribution permission

### 08 Cardiovascular Disease and Air Pollutants
- Source: Publicly available or published observational dataset cited in the manuscript
- Redistribution status: To verify
- Required action: Confirm the data provider, access location or identifier, retrieval date, terms of use, and whether raw or processed data may be redistributed

### 09 Damped Oscillator
- Source: Author-generated damped harmonic oscillator simulation
- Redistribution status: Expected to be redistributable
- Preserve: governing equation, coefficients, initial conditions, time interval, solver, and code version

## Software and package licensing

The CIRN code repository remains under pre-release review. Before public release:

- confirm software ownership and licensing with the coauthors and relevant institutional office;
- add an approved `LICENSE` file to GitHub;
- verify the licensing of bundled JavaScript, CSS, icons, images, and Shiny assets;
- confirm that example datasets bundled with CIRN Studio may be redistributed.

## Funding acknowledgment

The underlying dissertation work was supported by the Department of Science and Technology--Science Education Institute through the Accelerated Science and Technology Human Resource Development Program graduate scholarship and dissertation allowance awarded to G.M.E.

## Final public-release checklist

- [ ] Verify all third-party dataset licenses and terms of use
- [ ] Remove restricted raw data when redistribution is not permitted
- [ ] Add source citations and persistent identifiers
- [ ] Confirm contributor permissions
- [ ] Add approved software and data licenses
- [ ] Remove credentials, personal paths, and confidential information
- [ ] Confirm the exact GitHub release and commit supporting the article
- [ ] Freeze the approved OSF record or registration
- [ ] Add the DOI and formal dataset citation to the manuscript
