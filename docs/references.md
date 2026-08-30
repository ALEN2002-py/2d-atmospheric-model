# References

**Superseded as the primary bibliography source (August 2026).** The sources
Dr. Clancy provided directly (`docs/thesis/*.pdf`) have since been read in
full and page-cited in `docs/thesis/reference_notes/` (indexed by
`00_manifest.md`) — that folder is now the authoritative, verified reference
library, and Chapters 1–2 (`docs/thesis/01_introduction.md`,
`02_theory.md`) cite it directly with real page numbers. This file is kept
as a lighter-weight category-organised overview and for a handful of
standard technical citations (e.g. the original GMRES paper) that fall
outside Colm's provided set; where the two disagree, `reference_notes/`
is correct.

Working bibliography for the thesis. Organised by category, matching where
each source is first cited in `docs/thesis/`. Author–year in-text style is
used in the current markdown drafts; convert to the numeric `natbib`
(`apsrev`) style required by the LaTeX template as a single find-and-replace
pass once the chapters are ported into `thesis.tex`.

**Verification note:** entries marked ✓ were confirmed against a live source
this session. The rest are drawn from high-confidence recall (well-known,
widely-cited works) but have **not** been independently checked here —
volume/issue/page numbers in particular should be verified against the
publisher record (or Google Scholar / the journal DOI) before they go in the
final submitted bibliography. Treat this file as a working draft, not a
citation-ready reference list.

---

## Benchmark and primary source papers (already in use)

1. Giraldo, F.X., Restelli, M. (2008). A study of spectral element and
   discontinuous Galerkin methods for the Navier–Stokes equations in
   nonhydrostatic mesoscale atmospheric modeling: Equation sets and test
   cases. *Journal of Computational Physics*, 227, 3849–3877.
   DOI: 10.1016/j.jcp.2007.12.009

2. Pudykiewicz, J.A., Clancy, C. (2022). Convection experiments with the
   exponential time integration scheme. *Journal of Computational Physics*,
   449, 110803. DOI: 10.1016/j.jcp.2021.110803

3. Robert, A. (1993). Bubble convection experiments with a semi-implicit
   formulation of the Euler equations. *Journal of the Atmospheric
   Sciences*, 50(13), 1865–1873.

4. Straka, J.M., Wilhelmson, R.B., Wicker, L.J., Anderson, J.R.,
   Droegemeier, K.K. (1993). Numerical solutions of a nonlinear density
   current: a benchmark solution and comparisons. *International Journal
   for Numerical Methods in Fluids*, 17, 1–22.

5. Klemp, J.B., Wilhelmson, R.B. (1978). The simulation of
   three-dimensional convective storm dynamics. *Journal of the Atmospheric
   Sciences*, 35, 1070–1096.

## General textbooks (Theory, Section 2.1.1–2.1.2)

6. Holton, J.R., Hakim, G.J. (2013). *An Introduction to Dynamic
   Meteorology*, 5th ed. Academic Press. — standard graduate atmospheric
   dynamics textbook; general derivation of the governing equations,
   thermodynamic relations, and the Brunt–Väisälä frequency.

7. Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and
   Predictability*. Cambridge University Press.

8. Durran, D.R. (2010). *Numerical Methods for Fluid Dynamics: With
   Applications to Geophysics*, 2nd ed. Springer. — general treatment of
   time-integration schemes, stability theory, and diffusion in
   atmospheric-modelling numerics; used throughout Section 2.1.5–2.1.7.

## Numerical analysis / iterative & Krylov methods

9. ✓ LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and
   Partial Differential Equations: Steady-State and Time-Dependent
   Problems*. SIAM. — general finite-difference/stability-theory textbook
   (von Neumann analysis, Ch. 9).

10. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
    SIAM. — textbook treatment of Krylov subspace methods, incl. GMRES.

11. Saad, Y., Schultz, M.H. (1986). GMRES: A generalized minimal residual
    algorithm for solving nonsymmetric linear systems. *SIAM Journal on
    Scientific and Statistical Computing*, 7(3), 856–869. — the original
    GMRES paper; cited wherever the SI/SI2 linear solver is introduced.

12. Niesen, J., Wright, W.M. (2012). Algorithm 919: A Krylov subspace
    algorithm for evaluating the φ-functions appearing in exponential
    integrators. *ACM Transactions on Mathematical Software*, 38(3),
    Article 22. — the `phipm` algorithm used for the EPI schemes.

13. Hochbruck, M., Ostermann, A. (2010). Exponential integrators. *Acta
    Numerica*, 19, 209–286. — the standard general (non-atmospheric-specific)
    review of exponential integrator theory and phi-functions.

## Time-integration strategy in NWP — foundational and review papers

14. ✓ Mengaldo, G., Wyszogrodzki, A., Diamantakis, M., Lock, S.-J., Giraldo,
    F.X., Wedi, N.P. (2019). Current and emerging time-integration
    strategies in global numerical weather and climate prediction.
    *Archives of Computational Methods in Engineering*, 26, 663–684.
    DOI: 10.1007/s11831-018-9261-8 — the broad review Dr. Clancy pointed to
    directly; cited in Ch. 1 (§1.2.3) and Ch. 2 (§2.1.4) to situate this
    dissertation's scheme comparison within the wider NWP literature.

15. Robert, A. (1969). The integration of a spectral model of the
    atmosphere by the implicit method. In *Proceedings of the WMO/IUGG
    Symposium on NWP*, Tokyo, VII.19–VII.24. — the original semi-implicit
    scheme for NWP, cited in §2.1.5.

## Time filtering

16. Robert, A. (1966). The integration of a low order spectral form of the
    primitive meteorological equations. *Journal of the Meteorological
    Society of Japan*, 44, 237–245.

17. Asselin, R. (1972). Frequency filter for time integrations. *Monthly
    Weather Review*, 100(6), 487–490. — together with Robert (1966), the
    origin of the Robert–Asselin time filter used for CTCS and SI2 (§2.1.6).

## Von Neumann stability analysis — origin

18. Charney, J.G., Fjørtoft, R., von Neumann, J. (1950). Numerical
    integration of the barotropic vorticity equation. *Tellus*, 2(4),
    237–254. — the founding paper that introduced von Neumann stability
    analysis into numerical weather prediction; cited in §2.1.7.

## Diffusion / filtering in atmospheric models

19. Jablonowski, C., Williamson, D.L. (2011). The pros and cons of
    diffusion, filters and fixers in atmospheric general circulation
    models. In *Numerical Techniques for Global Atmospheric Models*,
    Lecture Notes in Computational Science and Engineering, vol. 80,
    Springer, Ch. 13.

---

## To add later (topics not yet covered above)

- A reference for the specific hyperdiffusion (∇²/∇⁴/∇⁸) formulation used
  in `dynamics.py`, if drawn from a specific paper rather than standard
  practice.
- A reference for the Shapiro filter (Shapiro, R., 1970 — "Smoothing,
  filtering, and boundary effects," *Reviews of Geophysics*, 8(2),
  359–387 — unverified, check before use).
- Any GMRES preconditioning literature, if the "unpreconditioned GMRES"
  caveat in the RK4-vs-SI/SI2 result (Chapter 4) is expanded on.
