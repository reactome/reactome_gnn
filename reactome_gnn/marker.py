from reactome2py import analysis


class Marker:
    def __init__(self, marker_list, p_value):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

    def enrichment_analysis(self):
        """
        Enrichment analysis performed on all the pathways. First all the hit pathways are obtained.
        Then, it is determined which of them are significant (p_value < threshold).
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
