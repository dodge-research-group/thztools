---
title: 'THzTools: data analysis software for terahertz time-domain spectroscopy'

tags:
  - Python
  - terahertz spectroscopy
  - time-domain spectroscopy
  - data analysis
  - parameter estimation
  - maximum-likelihood analysis

authors:
  - name: Jonathan Posada Loaiza
    orcid: 0009-0004-4443-6161
    equal-contrib: true
    affiliation: 1
  - name: Santiago Higuera-Quintero
    orcid: 0000-0001-6905-4662
    equal-contrib: true
    affiliation: 1
  - name: Alireza Noori
    orcid: 0009-0009-1091-0802
    equal-contrib: true
    affiliation: 1
  - name: Laleh Mohtashemi
    orcid: 0000-0002-2706-857X
    equal-contrib: true
    affiliation: 1
  - name: R. P. Hall
    orcid: 0009-0004-1137-3651
    affiliation: 1
  - name: Naod Ayalew Yimam
    orcid: 0009-0007-7522-5027
    affiliation: 1
  - name: J. Steven Dodge
    orcid: 0000-0003-3219-7314
    corresponding: true
    affiliation: 1

affiliations:
 - name: Department of Physics, Simon Fraser University, Canada
   index: 1

date: 4 May 2024

bibliography: paper.bib
---
# Summary
Terahertz time-domain spectroscopy (THz-TDS) uses short electromagnetic pulses
to probe the electromagnetic response of matter over frequencies from about
0.1 THz to about 10 THz, where
1 THz = 10^12^ Hz [@neu2018a]. A typical measurement
compares two electromagnetic pulses, one of which has interacted with a material
and another which has not. The pulses are measured as a function of time, but
the electromagnetic properties of matter are most naturally described as a
function of frequency, so statistical signal processing techniques must be used
to relate the time-domain measurements to the frequency-domain properties of
interest. The `THzTools` package provides an interface for implementing maximum-
likelihood methods for THz-TDS analysis, described
previously [@mohtashemi2021].

# Statement of need

In the conventional approach to THz-TDS analysis [@neu2018a], one transforms
the time-domain measurements into the frequency domain for further analysis.
This approach has well-known problems, however, which can be resolved by
using a maximum-likelihood estimation procedure in the time 
domain [@mohtashemi2021]. To support this mode of analysis, the `THzTools`
package provides functionality and documentation that are unavailable in
existing THz-TDS analysis software [@peretti2019; @tayvah2021; @lee2023]. It
provides functions to simulate THz-TDS measurements, apply a frequency response
function to a THz-TDS waveform, characterize the noise of a THz-TDS system, and
fit a parameterized frequency response function to a pair of input and output
waveforms.

We developed some of the functionality of `THzTools` in an earlier MATLAB
implementation, which remains undocumented [@dodge2021]. After translating
this codebase to Python, we introduced new functionality, optimized the code for
speed and efficiency, and revised the user interface to make it easier to use.

# Acknowledgements
J. P. L. and S. H.-G. acknowledge support from Mitacs Globalink Research
Internship Program. A. N. acknowledges support from the Natural Sciences and
Engineering Research Council of Canada Collaborative Research and Training
Experience (NSERC CREATE) Program in Quantum Computing. J. S. D. acknowledges
support from the Natural Sciences and Engineering Research Council of Canada
(NSERC).

# References