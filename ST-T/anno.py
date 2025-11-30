import wfdb

def list_edb_annotations():
    records = wfdb.get_record_list('edb')
    annotations = {}
    for record in records:
        ann = wfdb.rdann(f'st-t/{record}', 'atr')
        annotations[record] = list(zip(ann.symbol, ann.aux_note))
    return annotations

edb_annotations = list_edb_annotations()
print(edb_annotations)
