import pandas
pandas.set_option('display.notebook_repr_html', True)

def _repr_latex_(self):
    return r"""
    \begin{center}
    {%s}
    \end{center}
    """ % self.to_latex()

pandas.DataFrame._repr_latex_ = _repr_latex_  # monkey patch pandas DataFrame

pandas.set_option('display.max_columns', 6)
pandas.set_option('display.precision', 2)
pandas.set_option('display.max_colwidth', 12)
