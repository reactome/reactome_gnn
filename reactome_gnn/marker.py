from reactome2py import analysis


class Marker:
    """
    A class that performs enrichment analysis results for a certain
    set of markers and stores the results.

    Attributes
    ----------
    markers : str
        A coma-separated string of markers for which the enrichment
        analysis is performed
    p_value : float
        Threshold for p-value to determine significant pathways
    result : dict
        Results of the enrichment analysis
    """

    def __init__(self, marker_list, p_value):
        """
        Parameters
        ----------
        markers : str
            A coma-separated string of markers for which the enrichment
            analysis is performed
        p_value : float
            Threshold for p-value to determine significant pathways
        """
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

    def enrichment_analysis(self):
        """Enrichment analysis performed on all the pathways.
        
        First all the hit pathways are obtained. Then, it is determined
        which of them are significant (p_value < threshold).

        Returns
        -------
        dict
            Dictionary of significant pathways, where stids are keys
            and the values stored are p_value and significance of
            each pathway
        """
        result = analysis.identifiers(ids=self.markers, interactors=False, page_size='1', page='1',
                                      species='Homo Sapiens', sort_by='ENTITIES_FDR', order='ASC',
                                      resource='TOTAL', p_value='1', include_disease=False, min_entities=None,
                                      max_entities=None, projection=True)
        token = result['summary']['token']
        token_result = analysis.token(token, species='Homo sapiens', page_size='-1', page='-1', sort_by='ENTITIES_FDR',
                                      order='ASC', resource='TOTAL', p_value='1', include_disease=False,
                                      min_entities=None, max_entities=None)
        info = [(p['stId'], p['entities']['pValue']) for p in token_result['pathways']]
        pathway_significance = {}
        for stid, p_val in info:
            significance = 'significant' if p_val < self.p_value else 'non-significant'
            pathway_significance[stid] = {'p_value': round(p_val, 4), 'significance': significance}
        return pathway_significance
