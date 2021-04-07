from typing import Optional

import hail as hl
import hail.expr.aggregators as agg
from hail.expr import (ArrayNumericExpression, BooleanExpression,
                       CallExpression, Float64Expression, analyze, expr_array,
                       expr_call, expr_float64, matrix_table_source)
from hail.linalg import BlockMatrix
from hail.table import Table
from hail.typecheck import enumeration, nullable, numeric, typecheck

from ..pca import hwe_normalized_pca


def _bad_mu(mu: Float64Expression, maf: float) -> BooleanExpression:
    """Check if computed value for estimated individual-specific allele
    frequency (mu) is not valid for estimating relatedness.

    Parameters
    ----------
    mu : :class:`.Float64Expression`
        Estimated individual-specific allele frequency.
    maf : :obj:`float`
        Minimum individual-specific minor allele frequency.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `mu` is not valid for relatedness estimation, else ``False``.
    """
    return (mu <= maf) | (mu >= (1.0 - maf)) | (mu <= 0.0) | (mu >= 1.0)


def _bad_gt(gt: Float64Expression) -> BooleanExpression:
    """Check if genotype value is not valid.

    Parameters
    ----------
    gt : :class:`.Float64Expression`
        Allele count.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `gt` is not 0, 1, or 2. Else ``False``.
    """
    return (gt != 0.0) & (gt != 1.0) & (gt != 2.0)


def _gram(M: BlockMatrix) -> BlockMatrix:
    """Compute Gram matrix, `M.T @ M`.

    Parameters
    ----------
    M : :class:`.BlockMatrix`

    Returns
    -------
    :class:`.BlockMatrix`
        `M.T @ M`
    """
    return M.T @ M


def _AtB_plus_BtA(A: BlockMatrix, B: BlockMatrix) -> BlockMatrix:
    """Compute `(A.T @ B) + (B.T @ A)`, used in estimating IBD0 (k0).

    Parameters
    ----------
    A : :class:`.BlockMatrix`
    B : :class:`.BlockMatrix`

    Returns
    -------
    :class:`.BlockMatrix`
        `(A.T @ B) + (B.T @ A)`
    """
    temp = (A.T @ B).persist()
    return temp + temp.T


@typecheck(call_expr=expr_call,
           min_individual_maf=numeric,
           k=nullable(int),
           scores_expr=nullable(expr_array(expr_float64)),
           min_kinship=nullable(numeric),
           statistics=enumeration('kin', 'kin2', 'kin20', 'all'),
           block_size=nullable(int),
           include_self_kinship=bool)
def pc_relate_2(call_expr: CallExpression,
                min_individual_maf: float,
                *,
                k: Optional[int] = None,
                scores_expr: Optional[ArrayNumericExpression] = None,
                min_kinship: Optional[float] = None,
                statistics: str = "all",
                block_size: Optional[int] = None,
                include_self_kinship: bool = False) -> Table:
    r"""Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/req_diploid_gt.rst

    Examples
    --------
    Estimate kinship, identity-by-descent two, identity-by-descent one, and
    identity-by-descent zero for every pair of samples, using a minimum minor
    allele frequency filter of 0.01 and 10 principal components to control
    for population structure.

    >>> rel = hl.pc_relate_2(dataset.GT, 0.01, k=10)

    Only compute the kinship statistic. This is more efficient than
    computing all statistics.

    >>> rel = hl.pc_relate_2(dataset.GT, 0.01, k=10, statistics='kin')

    Compute all statistics, excluding sample-pairs with kinship less
    than 0.1. This is more efficient than producing the full table and
    then filtering using :meth:`.Table.filter`.

    >>> rel = hl.pc_relate_2(dataset.GT, 0.01, k=10, min_kinship=0.1)

    One can also pass in pre-computed principal component scores.
    To produce the same results as in the previous example:

    >>> _, scores_table, _ = hl.hwe_normalized_pca(dataset.GT,
    ...                                      k=10,
    ...                                      compute_loadings=False)
    >>> rel = hl.pc_relate_2(dataset.GT,
    ...                    0.01,
    ...                    scores_expr=scores_table[dataset.col_key].scores,
    ...                    min_kinship=0.1)

    Notes
    -----
    The traditional estimator for kinship between a pair of individuals
    :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
    single-nucleotide variants, from a population with allele frequencies
    :math:`p_s`, is given by:

    .. math::

      \widehat{\phi_{ij}} \coloneqq
        \frac{1}{|S_{ij}|}
        \sum_{s \in S_{ij}}
          \frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}
                {4 \sum_{s \in S_{ij}} p_s (1 - p_s)}

    This estimator is true under the model that the sharing of common
    (relative to the population) alleles is not very informative to
    relatedness (because they're common) and the sharing of rare alleles
    suggests a recent common ancestor from which the allele was inherited by
    descent.

    When multiple ancestry groups are mixed in a sample, this model breaks
    down. Alleles that are rare in all but one ancestry group are treated as
    very informative to relatedness. However, these alleles are simply
    markers of the ancestry group. The PC-Relate method corrects for this
    situation and the related situation of admixed individuals.

    PC-Relate slightly modifies the usual estimator for relatedness:
    occurrences of population allele frequency are replaced with an
    "individual-specific allele frequency". This modification allows the
    method to correctly weight an allele according to an individual's unique
    ancestry profile.

    The "individual-specific allele frequency" at a given genetic locus is
    modeled by PC-Relate as a linear function of a sample's first ``k``
    principal component coordinates. As such, the efficacy of this method
    rests on two assumptions:

     - an individual's first `k` principal component coordinates fully
       describe their allele-frequency-relevant ancestry, and

     - the relationship between ancestry (as described by principal
       component coordinates) and population allele frequency is linear

    The estimators for kinship, and identity-by-descent zero, one, and two
    follow. Let:

     - :math:`S_{ij}` be the set of genetic loci at which both individuals
       :math:`i` and :math:`j` have a defined genotype

     - :math:`g_{is} \in {0, 1, 2}` be the number of alternate alleles that
       individual :math:`i` has at genetic locus :math:`s`

     - :math:`\widehat{\mu_{is}} \in [0, 1]` be the individual-specific allele
       frequency for individual :math:`i` at genetic locus :math:`s`

     - :math:`{\widehat{\sigma^2_{is}}} \coloneqq \widehat{\mu_{is}} (1 - \widehat{\mu_{is}})`,
       the binomial variance of :math:`\widehat{\mu_{is}}`

     - :math:`\widehat{\sigma_{is}} \coloneqq \sqrt{\widehat{\sigma^2_{is}}}`,
       the binomial standard deviation of :math:`\widehat{\mu_{is}}`

     - :math:`\text{IBS}^{(0)}_{ij} \coloneqq \sum_{s \in S_{ij}} \mathbb{1}_{||g_{is} - g_{js} = 2||}`,
       the number of genetic loci at which individuals :math:`i` and :math:`j`
       share no alleles

     - :math:`\widehat{f_i} \coloneqq 2 \widehat{\phi_{ii}} - 1`, the inbreeding
       coefficient for individual :math:`i`

     - :math:`g^D_{is}` be a dominance encoding of the genotype matrix, and
       :math:`X_{is}` be a normalized dominance-coded genotype matrix

    .. math::

        g^D_{is} \coloneqq
          \begin{cases}
            \widehat{\mu_{is}}     & g_{is} = 0 \\
            0                        & g_{is} = 1 \\
            1 - \widehat{\mu_{is}} & g_{is} = 2
          \end{cases}

        \qquad
        X_{is} \coloneqq g^D_{is} - \widehat{\sigma^2_{is}} (1 + \widehat{f_i})

    The estimator for kinship is given by:

    .. math::

      \widehat{\phi_{ij}} \coloneqq
        \frac{\sum_{s \in S_{ij}}(g - 2 \mu)_{is} (g - 2 \mu)_{js}}
              {4 * \sum_{s \in S_{ij}}
                            \widehat{\sigma_{is}} \widehat{\sigma_{js}}}

    The estimator for identity-by-descent two is given by:

    .. math::

      \widehat{k^{(2)}_{ij}} \coloneqq
        \frac{\sum_{s \in S_{ij}}X_{is} X_{js}}{\sum_{s \in S_{ij}}
          \widehat{\sigma^2_{is}} \widehat{\sigma^2_{js}}}

    The estimator for identity-by-descent zero is given by:

    .. math::

      \widehat{k^{(0)}_{ij}} \coloneqq
        \begin{cases}
          \frac{\text{IBS}^{(0)}_{ij}}
                {\sum_{s \in S_{ij}}
                       \widehat{\mu_{is}}^2(1 - \widehat{\mu_{js}})^2
                       + (1 - \widehat{\mu_{is}})^2\widehat{\mu_{js}}^2}
            & \widehat{\phi_{ij}} > 2^{-5/2} \\
          1 - 4 \widehat{\phi_{ij}} + k^{(2)}_{ij}
            & \widehat{\phi_{ij}} \le 2^{-5/2}
        \end{cases}

    The estimator for identity-by-descent one is given by:

    .. math::

      \widehat{k^{(1)}_{ij}} \coloneqq
        1 - \widehat{k^{(2)}_{ij}} - \widehat{k^{(0)}_{ij}}

    Note that, even if present, phase information is ignored by this method.

    The PC-Relate method is described in "Model-free Estimation of Recent
    Genetic Relatedness". Conomos MP, Reiner AP, Weir BS, Thornton TA. in
    American Journal of Human Genetics. 2016 Jan 7. The reference
    implementation is available in the `GENESIS Bioconductor package
    <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`_ .

    :func:`.pc_relate_2` differs from the reference implementation in a few
    ways:

     - if `k` is supplied, samples scores are computed via PCA on all samples,
       not a specified subset of genetically unrelated samples. The latter
       can be achieved by filtering samples, computing PCA variant loadings,
       and using these loadings to compute and pass in scores for all samples.

     - the estimators do not perform small sample correction

     - the algorithm does not provide an option to use population-wide
       allele frequency estimates

     - the algorithm does not provide an option to not use "overall
       standardization" (see R ``pcrelate`` documentation)

    Under the PC-Relate model, kinship, :math:`\phi_{ij}`, ranges from 0 to
    0.5, and is precisely half of the
    fraction-of-genetic-material-shared. Listed below are the statistics for
    a few pairings:

     - Monozygotic twins share all their genetic material so their kinship
       statistic is 0.5 in expection.

     - Parent-child and sibling pairs both have kinship 0.25 in expectation
       and are separated by the identity-by-descent-zero, :math:`k^{(2)}_{ij}`,
       statistic which is zero for parent-child pairs and 0.25 for sibling
       pairs.

     - Avuncular pairs and grand-parent/-child pairs both have kinship 0.125
       in expectation and both have identity-by-descent-zero 0.5 in expectation

     - "Third degree relatives" are those pairs sharing
       :math:`2^{-3} = 12.5 %` of their genetic material, the results of
       PCRelate are often too noisy to reliably distinguish these pairs from
       higher-degree-relative-pairs or unrelated pairs.

    Note that :math:`g_{is}` is the number of alternate alleles. Hence, for
    multi-allelic variants, a value of 2 may indicate two distinct alternative
    alleles rather than a homozygous variant genotype. To enforce the latter,
    either filter or split multi-allelic variants first.

    The resulting table has the first 3, 4, 5, or 6 fields below, depending on
    the `statistics` parameter:

     - `i` (``col_key.dtype``) -- First sample. (key field)
     - `j` (``col_key.dtype``) -- Second sample. (key field)
     - `kin` (:py:data:`.tfloat64`) -- Kinship estimate, :math:`\widehat{\phi_{ij}}`.
     - `ibd2` (:py:data:`.tfloat64`) -- IBD2 estimate, :math:`\widehat{k^{(2)}_{ij}}`.
     - `ibd0` (:py:data:`.tfloat64`) -- IBD0 estimate, :math:`\widehat{k^{(0)}_{ij}}`.
     - `ibd1` (:py:data:`.tfloat64`) -- IBD1 estimate, :math:`\widehat{k^{(1)}_{ij}}`.

    Here ``col_key`` refers to the column key of the source matrix table,
    and ``col_key.dtype`` is a struct containing the column key fields.

    There is one row for each pair of distinct samples (columns), where `i`
    corresponds to the column of smaller column index. In particular, if the
    same column key value exists for :math:`n` columns, then the resulting
    table will have :math:`\binom{n-1}{2}` rows with both key fields equal to
    that column key value. This may result in unexpected behavior in downstream
    processing.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    min_individual_maf : :obj:`float`
        The minimum individual-specific minor allele frequency.
        If either individual-specific minor allele frequency for a pair of
        individuals is below this threshold, then the variant will not
        be used to estimate relatedness for the pair.
    k : :obj:`int`, optional
        If set, `k` principal component scores are computed and used.
        Exactly one of `k` and `scores_expr` must be specified.
    scores_expr : :class:`.ArrayNumericExpression`, optional
        Column-indexed expression of principal component scores, with the same
        source as `call_expr`. All array values must have the same positive length,
        corresponding to the number of principal components, and all scores must
        be non-missing. Exactly one of `k` and `scores_expr` must be specified.
    min_kinship : :obj:`float`, optional
        If set, pairs of samples with kinship lower than `min_kinship` are excluded
        from the results.
    statistics : :class:`str`
        Set of statistics to compute.
        If ``'kin'``, only estimate the kinship statistic.
        If ``'kin2'``, estimate the above and IBD2.
        If ``'kin20'``, estimate the above and IBD0.
        If ``'all'``, estimate the above and IBD1.
    block_size : :obj:`int`, optional
        Block size of block matrices used in the algorithm.
        Default given by :meth:`.BlockMatrix.default_block_size`.
    include_self_kinship: :obj:`bool`
        If ``True``, include entries for an individual's estimated kinship with
        themselves. Defaults to ``False``.

    Returns
    -------
    :class:`.Table`
        A :class:`.Table` mapping pairs of samples to their pair-wise statistics.
    """
    assert (min_individual_maf >= 0.0 and min_individual_maf <= 1.0), \
        f"invalid argument: min_individual_maf={min_individual_maf}. " \
        f"Must have min_individual_maf on interval [0.0, 1.0]."
    mt = matrix_table_source('pc_relate_2/call_expr', call_expr)

    if k and scores_expr is None:
        _, scores, _ = hwe_normalized_pca(call_expr, k, compute_loadings=False)
        scores_expr = scores[mt.col_key].scores
    elif not k and scores_expr is not None:
        analyze('pc_relate_2/scores_expr', scores_expr, mt._col_indices)
    elif k and scores_expr is not None:
        raise ValueError("pc_relate_2: exactly one of 'k' and 'scores_expr' "
                         "must be set, found both")
    else:
        raise ValueError("pc_relate_2: exactly one of 'k' and 'scores_expr' "
                         "must be set, found neither")

    scores_table = mt.select_cols(__scores=scores_expr) \
        .key_cols_by().select_cols('__scores').cols()

    n_missing = scores_table.aggregate(agg.count_where(hl.is_missing(scores_table.__scores)))
    if n_missing > 0:
        raise ValueError(f'Found {n_missing} columns with missing scores array.')

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles()).unfilter_entries()
    mt = mt.annotate_rows(__mean_gt=agg.mean(mt.__gt))
    mean_imputed_gt = hl.or_else(hl.float64(mt.__gt), mt.__mean_gt)

    if not block_size:
        block_size = BlockMatrix.default_block_size()

    g_bm = BlockMatrix.from_entry_expr(mean_imputed_gt, block_size=block_size).persist()

    pcs = hl.nd.array(scores_table.collect(_localize=False).map(lambda x: x.__scores))

    # Concat array of ones (intercept) with PCs, do QR
    v = hl.nd.concatenate([hl.nd.ones((pcs.shape[0], 1)), pcs], axis=1)._persist()
    q, r = hl.nd.qr(v, mode='reduced')

    # Compute beta and mu
    rinv_qt_bm = BlockMatrix.from_numpy(hl.eval(hl.nd.inv(r) @ q.T), block_size=block_size).persist()
    beta_bm = (rinv_qt_bm @ g_bm.T).persist()
    v_bm = BlockMatrix.from_numpy(hl.eval(v), block_size=block_size)
    mu_bm = (0.5 * (v_bm @ beta_bm).T)

    # Define NaN to use instead of missing, otherwise cannot go back to block matrix
    nan = hl.literal(0) / 0

    # Replace invalid values for g and mu w/ NaN
    g_mt = g_bm.to_matrix_table_row_major()
    g_mt = g_mt.annotate_entries(g=hl.if_else(_bad_gt(g_mt.element),
                                              nan,
                                              g_mt.element)).drop("element")
    pre_mu_mt = mu_bm.to_matrix_table_row_major()
    pre_mu_mt = pre_mu_mt.annotate_entries(pre_mu=hl.if_else(_bad_mu(pre_mu_mt.element, min_individual_maf),
                                                             nan,
                                                             pre_mu_mt.element)).drop("element")

    # Use bm_mt to store entries for g, pre_mu, mu, mu**2, (1-mu)**2, var, std_dev, and centered_af
    bm_mt = g_mt.annotate_entries(pre_mu=pre_mu_mt[g_mt.row_idx, g_mt.col_idx].pre_mu)
    bm_mt = bm_mt.annotate_entries(mu=hl.if_else(hl.is_nan(bm_mt.g) | hl.is_nan(bm_mt.pre_mu),
                                                 nan,
                                                 bm_mt.pre_mu))
    bm_mt = bm_mt.annotate_entries(mu2=hl.if_else(hl.is_nan(bm_mt.mu),
                                                  0.0,
                                                  bm_mt.mu ** 2),
                                   one_minus_mu2=hl.if_else(hl.is_nan(bm_mt.mu),
                                                            0.0,
                                                            (1.0 - bm_mt.mu) ** 2),
                                   variance=hl.if_else(hl.is_nan(bm_mt.mu),
                                                       0.0,
                                                       (bm_mt.mu * (1.0 - bm_mt.mu))),
                                   centered_af=hl.if_else(hl.is_nan(bm_mt.mu),
                                                          0.0,
                                                          (bm_mt.g / 2) - bm_mt.mu))
    bm_mt = bm_mt.annotate_entries(std_dev=hl.sqrt(bm_mt.variance))

    # Compute kinship (phi) estimate
    centered_af_bm = BlockMatrix.from_entry_expr(bm_mt.centered_af, block_size=block_size)
    std_dev_bm = BlockMatrix.from_entry_expr(bm_mt.std_dev, block_size=block_size)
    phi_bm = (_gram(centered_af_bm) / _gram(std_dev_bm)).persist()
    ht = phi_bm.entries().rename({'entry': 'kin'})
    ht = ht.annotate(k0=hl.missing(hl.tfloat64),
                     k1=hl.missing(hl.tfloat64),
                     k2=hl.missing(hl.tfloat64))

    if statistics is "kin2" or "kin20" or "all":
        # Create table w/ self-kinship (phi_ii) values
        phi_ii_ht = phi_bm.diagonal().entries().key_by("j").drop("i") \
            .rename({"j": "idx", "entry": "phi_ii"})

        # Annotate cols of bm_mt w/ self-kinship (phi_ii) and inbreeding coef (f_i)
        bm_mt = bm_mt.annotate_cols(phi_ii=phi_ii_ht[bm_mt.col_idx].phi_ii,
                                    f_i=(2.0 * phi_ii_ht[bm_mt.col_idx].phi_ii) - 1.0)

        # Create entries for dominance encoding of genotype matrix (gd and normalized_gd)
        bm_mt = bm_mt.annotate_entries(gd=hl.case()
                                       .when(hl.is_nan(bm_mt.mu), 0.0)
                                       .when(bm_mt.g == 0.0, bm_mt.mu)
                                       .when(bm_mt.g == 1.0, 0.0)
                                       .when(bm_mt.g == 2.0, 1 - bm_mt.mu)
                                       .default(nan))
        bm_mt = bm_mt.annotate_entries(normalized_gd=bm_mt.gd - bm_mt.variance * (1 + bm_mt.f_i))

        # Compute IBD2 (k2) estimate
        normalized_gd_bm = BlockMatrix.from_entry_expr(bm_mt.normalized_gd, block_size=block_size)
        variance_bm = BlockMatrix.from_entry_expr(bm_mt.variance, block_size=block_size)
        k2_bm = _gram(normalized_gd_bm) / _gram(variance_bm)
        ht = ht.annotate(k2=k2_bm.entries()[ht.i, ht.j].entry)

        if statistics is "kin20" or "all":
            # Compute IBS0
            bm_mt = bm_mt.annotate_entries(hom_alt=hl.if_else((hl.is_nan(bm_mt.mu) | (bm_mt.g != 2.0)),
                                                              0.0,
                                                              1.0),
                                           hom_ref=hl.if_else((hl.is_nan(bm_mt.mu) | (bm_mt.g != 0.0)),
                                                              0.0,
                                                              1.0))
            hom_alt_bm = BlockMatrix.from_entry_expr(bm_mt.hom_alt, block_size=block_size)
            hom_ref_bm = BlockMatrix.from_entry_expr(bm_mt.hom_ref, block_size=block_size)
            ibs0_bm = _AtB_plus_BtA(hom_alt_bm, hom_ref_bm)

            _k0_cutoff = 2.0 ** (-5.0 / 2.0)

            # Compute IBD0 (k0) estimates for when phi > _k0_cutoff
            mu2_bm = BlockMatrix.from_entry_expr(bm_mt.mu2, block_size=block_size)
            one_minus_mu2_bm = BlockMatrix.from_entry_expr(bm_mt.one_minus_mu2, block_size=block_size)
            k0_bm = ibs0_bm / _AtB_plus_BtA(mu2_bm, one_minus_mu2_bm)
            ht = ht.annotate(k0=k0_bm.entries()[ht.i, ht.j].entry)

            # Correct the IBD0 (k0) estimates for when phi <= _k0_cutoff
            ht = ht.annotate(k0=hl.if_else(ht.kin <= _k0_cutoff,
                                           (1.0 - (4.0 * ht.kin) + ht.k2),
                                           ht.k0))

            if statistics is "all":
                ht = ht.annotate(k1=1 - ht.k2 - ht.k0)

    ht = ht.rename({"k0": "ibd0", "k1": "ibd1", "k2": "ibd2"})

    if min_kinship:
        ht = ht.filter(ht.kin <= min_kinship, keep=False)

    if statistics is not "all":
        _fields_to_drop = {
            "kin": ["ibd0", "ibd1", "ibd2"],
            "kin2": ["ibd0", "ibd1"],
            "kin20": ["ibd1"]
        }
        ht = ht.drop(*_fields_to_drop[statistics])

    if not include_self_kinship:
        ht = ht.filter(ht.i == ht.j, keep=False)

    col_keys = hl.literal(mt.select_cols().key_cols_by().cols().collect(),
                          dtype=hl.tarray(mt.col_key.dtype))
    return ht.key_by(i=col_keys[hl.int32(ht.i)], j=col_keys[hl.int32(ht.j)])
