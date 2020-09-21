import json
import os
import warnings

import hail as hl
import pkg_resources
import requests
from hailtop.utils import retry_response_returning_functions

from . import lens
from ..matrixtable import matrix_table_type
from ..table import table_type
from ..typecheck import typecheck_method, oneof
from ..utils.java import Env


class DatasetVersion:
    """A `DatasetVersion` instance.

    To create a `DatasetVersion` instance with the url and version (e.g. GRCh37)
    of a particular dataset specified in JSON configuration file or a provided
    dict mapping dataset names to configurations. Facilitates creation of
    Dataset objects with `Dataset.from_name_and_json()` method by providing
    versions of datasets only if they are available in the specified region via
    the `get_region()` method.
    """
    @staticmethod
    def from_json(doc):
        """Create `DatasetVersion` object from dictionary.

        Parameters
        ----------
        doc : :obj:`dict`
            Dictionary containing url and version keys.

        Returns
        -------
        :class:`DatasetVersion`
        """
        assert 'url' in doc, doc
        assert 'version' in doc, doc
        return DatasetVersion(doc['url'],
                              doc['version'])

    @staticmethod
    def get_region(name, versions, region):
        """Get versions of a `Dataset` in the specified region, if they exist.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.
        versions : :obj:`dict`
            Dictionary containing url and version, where the value for url is
            a dictionary containing the regions 'us' and 'eu'.
        region : :obj: `str`
            GCP region from which to access data, available regions given in
            hl.experimental.DB._valid_regions, currently either 'us' or 'eu'.

        Returns
        -------
        available_versions : :obj:`list`
            List of available versions of a `Dataset` for region.
        """
        available_versions = []
        if region is not None:
            for version in versions:
                if version.in_region(name, region):
                    version.url = version.url[region]
                    available_versions.append(version)
        return available_versions

    def __init__(self, url, version):
        self.url = url
        self.version = version

    def in_region(self, name, region):
        """To check if a `DatasetVersion` object is accessible in the desired
        region.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.
        region : :obj: `str`
            GCP region from which to access data, available regions given in
            hl.experimental.DB._valid_regions, currently either 'us' or 'eu'.

        Returns
        -------
        valid_region : :obj:`bool`
            Whether or not the dataset exists in the specified region.
        """
        current_version = self.version
        available_regions = [k for k, v in self.url.items() if v is not None]
        valid_region = region in available_regions
        if not valid_region:
            message = f'dataset: \'{name}\', version: {current_version}, exists but is not yet ' \
                      f'available in \'{region}\' region bucket, ' \
                      f'currently available in {available_regions} region bucket(s).'
            warnings.warn(message, UserWarning, stacklevel=1)
        return valid_region

    def maybe_index(self, indexer_key_expr, all_matches):
        return hl.read_table(self.url)._maybe_flexindex_table_by_expr(
            indexer_key_expr, all_matches=all_matches)


class Dataset:
    """A `Dataset` instance.

    To create a `Dataset` object with name, description, url, key_properties, and
    versions specified in JSON configuration file or a provided dict mapping
    dataset names to configurations.
    """
    @staticmethod
    def from_name_and_json(name, doc, region, custom_config=False):
        """Create `Dataset` object from dictionary.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.
        doc : :obj:`dict`
            Dictionary containing dataset description, url, key_properties, and
            versions.
        region : :obj: `str`
            GCP region from which to access data, available regions given in
            hl.experimental.DB._valid_regions, currently either 'us' or 'eu'.
        custom_config : :obj: `bool`
            Boolean indicating whether or not dataset is from a `DB` object
            using a custom configuration or url. If `True`, method will not
            check for region.
        Returns
        -------
        :class:`Dataset`
            If versions exist for region returns a `Dataset` object, else None.
        """
        assert 'description' in doc, doc
        assert 'url' in doc, doc
        assert 'key_properties' in doc, doc
        assert 'versions' in doc, doc
        versions = [DatasetVersion.from_json(x) for x in doc['versions']]
        if not custom_config:
            versions = DatasetVersion.get_region(name, versions, region)
        if versions:
            return Dataset(name,
                           doc['description'],
                           doc['url'],
                           set(doc['key_properties']),
                           versions)

    def __init__(self, name, description, url, key_properties, versions):
        assert set(key_properties).issubset(DB._valid_key_properties)
        self.name = name
        self.description = description
        self.url = url
        self.key_properties = key_properties
        self.versions = versions

    def is_gene_keyed(self):
        """Check if `Dataset` is gene keyed.

        Returns
        -------
        :obj:`bool`
            Whether or not `Dataset` is gene keyed.
        """
        return 'gene' in self.key_properties

    def index_compatible_version(self, key_expr):
        # If not unique key then use all matches, otherwise give a single a value
        # Add documentation here soon
        all_matches = 'unique' not in self.key_properties
        compatible_indexed_values = [
            index
            for index in (version.maybe_index(key_expr, all_matches)
                          for version in self.versions)
            if index is not None]
        if len(compatible_indexed_values) == 0:
            raise ValueError(
                f'Could not find compatible version of {self.name} for user '
                f'dataset with key {key_expr.dtype}.')
        assert len(compatible_indexed_values) == 1, \
            f'{key_expr.dtype}, {self.name}, {compatible_indexed_values}'
        return compatible_indexed_values[0]


class DB:
    """An annotation database instance.

    This class facilitates the annotation of genetic datasets with variant
    annotations. It accepts either an HTTP(S) URL to an Annotation DB
    configuration or a python :obj:`dict` describing an Annotation DB
    configuration. User must specify the region ('us' or 'eu') in which the
    cluster is running if connecting to the default Hail Annotation DB.

    Examples
    --------
    Create an annotation database connecting to the default Hail Annotation DB:

    >>> db = hl.experimental.DB(region='us')
    >>> mt = db.annotate_rows_db(mt, 'gnomad_lof_metrics') # doctest: +SKIP
    """

    _valid_key_properties = {'gene', 'unique'}
    _valid_regions = {'us', 'eu'}

    def __init__(self,
                 *,
                 region=None,
                 url=None,
                 config=None):
        custom_config = config or url
        if region not in DB._valid_regions and not (config or url):
            raise ValueError(f'Specify valid region parameter, received: region={region}. '
                             f'Valid regions are {DB._valid_regions}.')
        if config is not None and url is not None:
            raise ValueError(f'Only specify one of the parameters url and config, '
                             f'received: url={url} and config={config}')
        if config is None:
            if url is None:
                config_path = pkg_resources.resource_filename(__name__, "annotation_db.json")
                assert os.path.exists(config_path), f'{config_path} does not exist'
                with open(config_path) as f:
                    config = json.load(f)
            else:
                response = retry_response_returning_functions(
                    requests.get, url)
                config = response.json()
            assert isinstance(config, dict)
        else:
            if not isinstance(config, dict):
                raise ValueError(f'expected a dict mapping dataset names to '
                                 f'configurations, but found {config}')
        self.region = region
        self.url = url
        self.config = config
        self.__by_name = {k: Dataset.from_name_and_json(k, v, region, custom_config)
                          for k, v in config.items()
                          if Dataset.from_name_and_json(k, v, region, custom_config) is not None}

    def available_databases(self):
        """Retrieve list of names of available databases.

        Returns
        -------
        :obj:`list`
            List of available databases.
        """
        return sorted(self.__by_name.keys())

    @staticmethod
    def _row_lens(rel):
        if isinstance(rel, hl.MatrixTable):
            return lens.MatrixRows(rel)
        elif isinstance(rel, hl.Table):
            return lens.TableRows(rel)
        else:
            raise ValueError(
                'annotation database can only annotate Hail MatrixTable or Table')

    def dataset_by_name(self, name):
        """Retrieve Dataset object by name.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.

        Returns
        -------
        :class:`Dataset`
            Dataset object.
        """
        if name not in self.__by_name:
            raise ValueError(
                f'{name} not found in annotation database, you may list all '
                f'known dataset names with available_databases()')
        return self.__by_name[name]

    def _annotate_gene_name(self, rel):
        gene_field = Env.get_uid()
        gencode = self.__by_name['gencode'].index_compatible_version(rel.key)
        return gene_field, rel.annotate(**{gene_field: gencode.gene_name})

    def _check_availability(self, names):
        """Check if datasets given in `names` are available in the annotation
        database instance.

        Parameters
        ----------
        names : :obj:`iterable`
            Tuple or list of names to check.
        """
        unavailable = [x for x in names if x not in self.__by_name.keys()]
        if unavailable:
            raise ValueError(f'datasets: {unavailable} not available in the {self.region} region.')

    @typecheck_method(rel=oneof(table_type, matrix_table_type), names=str)
    def annotate_rows_db(self, rel, *names):
        """Add annotations from datasets specified by name.

        List datasets with at :meth:`.available_databases`. An interactive query
        builder is available in the
        `Hail Annotation Database documentation </docs/0.2/annotation_database_ui.html>`_.

        Examples
        --------
        Annotate a matrix table with the `gnomad_lof_metrics`:

        >>> db = hl.experimental.DB(region='us')
        >>> mt = db.annotate_rows_db(mt, 'gnomad_lof_metrics') # doctest: +SKIP

        Annotate a table with `clinvar_gene_summary`, `CADD`, and `DANN`:

        >>> db = hl.experimental.DB(region='us')
        >>> mt = db.annotate_rows_db(mt, 'clinvar_gene_summary', 'CADD', 'DANN') # doctest: +SKIP

        Notes
        -----

        If a dataset is gene-keyed, the annotation will be a dictionary mapping
        from gene name to the annotation value. There will be one entry for each
        gene overlapping the given locus.

        If a dataset does not have unique rows for each key (consider the
        `gencode` genes, which may overlap; and `clinvar_variant_summary`, which
        contains many overlapping multiple nucleotide variants), then the result
        will be an array of annotation values, one for each row.

        Parameters
        ----------
        rel : :class:`.MatrixTable` or :class:`.Table`
            The relational object to which to add annotations.
        names : varargs of :obj:`str`
            The names of the datasets with which to annotate `rel`.

        Returns
        -------
        :class:`.MatrixTable` or :class:`.Table
            The original dataset with new annotations added.
        """
        rel = self._row_lens(rel)
        if len(set(names)) != len(names):
            raise ValueError(
                f'cannot annotate same dataset twice, please remove duplicates from: {names}')
        self._check_availability(names)
        datasets = [self.dataset_by_name(name) for name in names]
        if any(dataset.is_gene_keyed() for dataset in datasets):
            gene_field, rel = self._annotate_gene_name(rel)
        else:
            gene_field = None
        for dataset in datasets:
            if dataset.is_gene_keyed():
                genes = rel.select(gene_field).explode(gene_field)
                genes = genes.annotate(**{
                    dataset.name: dataset.index_compatible_version(genes[gene_field])})
                genes = genes.group_by(*genes.key)\
                             .aggregate(**{
                                 dataset.name: hl.dict(
                                     hl.agg.filter(hl.is_defined(genes[dataset.name]),
                                                   hl.agg.collect((genes[gene_field],
                                                                   genes[dataset.name]))))})
                rel = rel.annotate(**{dataset.name: genes.index(rel.key)[dataset.name]})
            else:
                indexed_value = dataset.index_compatible_version(rel.key)
                if isinstance(indexed_value.dtype, hl.tstruct) and len(indexed_value.dtype) == 0:
                    indexed_value = hl.is_defined(indexed_value)
                rel = rel.annotate(**{dataset.name: indexed_value})
        if gene_field:
            rel = rel.drop(gene_field)
        return rel.unlens()
