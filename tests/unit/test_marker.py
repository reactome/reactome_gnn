from reactome_gnn import marker

marker_list = ['RAS', 'MAP', 'IL10', 'EGF', 'EGFR', 'STAT']
ea_result = marker.Marker(marker_list, p_value=0.05)

def test_markers():
    assert isinstance(ea_result.markers, str)

def test_pval():
    assert isinstance(ea_result.p_value, float)

def test_result():
    assert isinstance(ea_result.result, dict)

def test_result_keys():
    it = iter(ea_result.result.items())
    key = next(it)[0]
    assert 'R-HSA' in key

def test_result_values():
    it = iter(ea_result.result.items())
    val_keys = tuple(next(it)[1].keys())
    assert ('p_value', 'significance') == val_keys
