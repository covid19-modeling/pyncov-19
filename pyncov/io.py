# -*- coding: utf-8 -*-
import os.path
import csv
from ssl import _create_unverified_context
from urllib.request import urlopen
from io import StringIO

DEFAULT_URL_PARAMETERS = 'https://raw.githubusercontent.com/covid19-modeling/forecasts/master/results/latest' \
                         '/parameters.csv'


def _unverified_stream_url(url):
    response = urlopen(url, context=_create_unverified_context())
    data = response.read()
    return StringIO(data.decode('utf-8'))


def _filter_csv_rows(stream, key_values, **kwargs):
    reader = csv.DictReader(stream, **kwargs)
    selected_rows = []
    for row in reader:
        if key_values is not None and len(key_values) > 0:
            for k, v in key_values:
                if row[k] == v:
                    selected_rows.append(row)
        else:
            selected_rows.append(row)
    return selected_rows


def read_csv(file_or_url, key_values=None, quotechar='"', delimiter=',',
             quoting=csv.QUOTE_ALL, skipinitialspace=True):

    stream = None
    try:
        if os.path.isfile(file_or_url):
            stream = open(file_or_url, encoding='utf-8')
        else:
            stream = _unverified_stream_url(file_or_url)

        rows = _filter_csv_rows(stream, key_values, quotechar=quotechar, delimiter=delimiter,
                                quoting=quoting, skipinitialspace=skipinitialspace)
    finally:
        if stream is not None:
            stream.close()
    return rows


def get_trained_params(id, url=DEFAULT_URL_PARAMETERS):
    d = read_csv(url, key_values=[('id', id)])[0]
    return [float(d[k]) for k in d.keys() if k.startswith('param')]
