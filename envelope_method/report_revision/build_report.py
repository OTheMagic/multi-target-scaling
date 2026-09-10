"""Assemble the revised research report, retaining the audited detailed proofs."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

original=(HERE/'original_signed_envelope.tex').read_text(encoding='utf-8')
appendix=original.split('\\section{Setup and the conformal rank}',1)[1].split('\\section{Reproducibility and comparison protocol}',1)[0]
appendix='\\section{Setup and the conformal rank}'+appendix

# Explicitly connect the old GWC definition to the domain used by the proof.
needle='It does not compare signed residuals to capped or shifted residuals. Assume\n$s_j>0$.'
replacement=r'''It does not compare signed residuals to capped or shifted residuals. Assume
$s_j>0$.

The mathematical old GWC transformation is the same exact supremum
$\max_j\psi_{j,[0,\infty)}(t_j)$ as \eqref{eq:gwcscore}. Its implementation
evaluates the endpoint at zero, the limit at infinity, and the stationary
point. For $t_j>m_j$, an admissible stationary point is exactly the maximum
in \eqref{eq:critical}; for $t_j<m_j$ the stationary point is a minimum and
cannot exceed both endpoints. Thus, in exact arithmetic and without jitter,
$G^{\mathrm{old}}=G$ for the same nonnegative scores and rank.
The implementation's nonnegative jitter can only enlarge the GWC quantile;
the theoretical comparison below is made with the unperturbed closed rules.
'''
assert needle in appendix
appendix=appendix.replace(needle,replacement)

# Generalize the off-coordinate scale to a mean clipped by the GWC interval.
start=appendix.index('If the old box contains the mean, choose each off-coordinate cell')
end=appendix.index('\\begin{lemma}[Pointwise score domination]',start)
appendix=appendix[:start]+r'''Let $p_\ell=\operatorname{proj}_{G^{\mathrm{old}}_\ell}(m_\ell)$ and
$\bar s_\ell=\sigma_\ell(p_\ell)=\inf_{z\in G^{\mathrm{old}}_\ell}\sigma_\ell(z)$.
If $m_\ell\in G^{\mathrm{old}}_\ell$, then $p_\ell=m_\ell$ and
$\bar s_\ell=s_\ell$. Otherwise $\bar s_\ell$ is the minimum scale on the
clipped domain, which need not equal $s_\ell$.
Choose each off-coordinate old cell to contain $p_\ell$. Whenever the old
mean cell intersects GWC it has this property after clipping. Its minimum
scale is therefore $\bar s_\ell$. The old row score is
\begin{equation}\label{eq:oldrow}
 \Phi^{\mathrm{old}}_{j,k}(\tvec)=
 \max\left\{\frac{t_j}{r_j(k)}-c_j,
             \max_{\ell\ne j}\left(\frac{t_\ell}{\bar s_\ell}-c_\ell\right)\right\}.
\end{equation}
Its quantile and clipped coordinate pieces define $L_j^{\mathrm{old}}$ as
in \eqref{eq:surfaceq}--\eqref{eq:L}, using the old cells. If the old mean
cell is unavailable, the manuscript shortcut returns its GWC box.

'''+appendix[end:]
appendix=appendix.replace('$\\sigma_\\ell(z)\\ge s_\\ell$', '$\\sigma_\\ell(z)\\ge\\bar s_\\ell$')
appendix=appendix.replace('$F_\\ell(t_\\ell,z)\\le t_\\ell/s_\\ell-c_\\ell$', '$F_\\ell(t_\\ell,z)\\le t_\\ell/\\bar s_\\ell-c_\\ell$')

refs=r'''
\begin{thebibliography}{9}
\bibitem{shafer2008} G. Shafer and V. Vovk (2008).
A Tutorial on Conformal Prediction. \emph{Journal of Machine Learning Research}
9:371--421. \url{https://jmlr.org/papers/v9/shafer08a.html}.
\bibitem{romano2019} Y. Romano, E. Patterson, and E. J. Cand\`es (2019).
Conformalized Quantile Regression. \emph{Advances in Neural Information
Processing Systems} 32.
\url{https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html}.
\end{thebibliography}
\end{document}
'''
body=(HERE/'report_main.tex').read_text(encoding='utf-8')
body=body.replace('% AUDITED_APPENDIX_INSERTION',appendix+refs)
body=body.replace('report_figures/', 'figures/report_revision/')
# report_main.tex compiles locally; the assembled parent note uses its own paths.
body=body.replace('\\graphicspath{{figures/}}', '\\graphicspath{{figures/report_revision/}}')
body=body.replace('\\input{table_n80.tex}', '\\input{report_revision/table_n80.tex}')
body=body.replace('\\input{table_volume.tex}', '\\input{report_revision/table_volume.tex}')
target=ROOT/'envelope_method/signed_envelope.tex'
target.write_text(body,encoding='utf-8')
dest=ROOT/'output/pdf/envelope_shortcut_report'
dest.mkdir(parents=True,exist_ok=True)
(dest/target.name).write_text(body.replace('figures/report_revision/', 'figures/'),encoding='utf-8')
shutil.copytree(ROOT/'envelope_method/figures/report_revision',dest/'figures',dirs_exist_ok=True)
(dest/'report_revision').mkdir(exist_ok=True)
for name in ['table_n80.tex','table_volume.tex']:
    shutil.copy2(HERE/name,dest/'report_revision'/name)
print('Assembled revised TeX and portable figure/source directory:',dest)
