#!/usr/bin/env python3
"""Independent black-box acceptance. No implementation imports or package assumptions."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact')
    parser.add_argument('--report')
    args = parser.parse_args()
    root = Path(args.artifact).resolve()
    script = root if root.is_file() else root / 'booking_conflicts.py'
    rows = []
    if not script.is_file():
        result = {'status': 'unavailable', 'reason': 'booking_conflicts.py missing', 'artifact': str(script), 'passed': 0, 'total': 0, 'cases': []}
    else:
        def b(i, room='A', start=0, end=10):
            return dict(id=i, room=room, start=start, end=end)
        def case(name, data, expected=None, invalid=False, raw=False, preserve=False):
            with tempfile.TemporaryDirectory(prefix='booking-acceptance-') as tmp:
                p=Path(tmp); inp=p/'input.json'; out=p/'output.json'; unrelated=p/'unrelated'; unrelated.mkdir()
                inp.write_text(data if raw else json.dumps(data), encoding='utf-8')
                sentinel=b'KEEP EXISTING OUTPUT\n'
                if preserve: out.write_bytes(sentinel)
                try:
                    cp=subprocess.run([sys.executable,str(script),'--input',str(inp),'--output',str(out)],cwd=unrelated,capture_output=True,text=True,timeout=10)
                    actual=None
                    if invalid:
                        passed=cp.returncode != 0 and bool(cp.stderr.strip()) and (out.read_bytes()==sentinel if preserve and out.exists() else not out.exists() and not preserve)
                    else:
                        try: actual=json.loads(out.read_text())
                        except (OSError, ValueError): pass
                        passed=cp.returncode==0 and actual=={'conflicts':expected}
                    rows.append({'name':name,'passed':passed,'exit_code':cp.returncode,'expected':None if invalid else {'conflicts':expected},'actual':actual,'stderr':cp.stderr[-1200:],'stdout':cp.stdout[-400:]})
                except Exception as e:
                    rows.append({'name':name,'passed':False,'error':f'{type(e).__name__}: {e}'})
        case('empty', [], [])
        case('one', [b('a')], [])
        case('simple_overlap', [b('z'), b('a',start=5,end=15)], [['a','z']])
        case('touching_endpoints', [b('a'),b('b',start=10,end=20)], [])
        case('different_rooms', [b('a'),b('b',room='B')], [])
        case('room_case_sensitive', [b('a',room='A'),b('b',room='a')], [])
        case('containment', [b('outer',start=0,end=1440),b('inner',start=5,end=6)], [['inner','outer']])
        case('identical_intervals', [b('b'),b('a')], [['a','b']])
        case('all_pairs_sorted', [b('z'),b('m'),b('a')], [['a','m'],['a','z'],['m','z']])
        sample=[b('a',start=60,end=120),b('b',start=120,end=180),b('c',start=90,end=150),b('d',room='B',start=60,end=180)]
        case('mixed_sample', sample, [['a','c'],['b','c']])
        case('reordered_sample', list(reversed(sample)), [['a','c'],['b','c']])
        case('boundary_minutes', [b('a',start=0,end=1),b('b',start=1439,end=1440)], [])
        case('unicode_ids', [b('é'),b('a')], [['a','é']])
        case('extra_fields', [dict(b('a'),note='ignored'),b('b')], [['a','b']])
        invalids=[
            ('malformed_json','{',True), ('wrong_top_level',{},False),('null_top_level',None,False),
            ('nonobject_row',[1],False), ('missing_id',[{'room':'A','start':0,'end':1}],False),
            ('empty_id',[b('')],False), ('numeric_id',[b(7)],False), ('empty_room',[b('a',room='')],False),
            ('missing_room',[{'id':'a','start':0,'end':1}],False), ('null_room',[b('a',room=None)],False),
            ('start_bool',[b('a',start=True)],False), ('end_bool',[b('a',start=0,end=True)],False),
            ('start_float',[b('a',start=0.5)],False), ('start_string',[b('a',start='0')],False),
            ('missing_end',[{'id':'a','room':'A','start':0}],False), ('negative_start',[b('a',start=-1)],False),
            ('zero_interval',[b('a',start=10,end=10)],False), ('reversed_interval',[b('a',start=20,end=10)],False),
            ('end_above_day',[b('a',end=1441)],False), ('duplicate_same_room',[b('a'),b('a')],False),
            ('duplicate_across_rooms',[b('a'),b('a',room='B')],False)]
        for name,data,raw in invalids:
            case(name,data,invalid=True,raw=raw)
            case(name+'_preserves_output',data,invalid=True,raw=raw,preserve=True)
        result={'status':'passed' if all(r['passed'] for r in rows) else 'failed','artifact':str(script),'artifact_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'passed':sum(r['passed'] for r in rows),'total':len(rows),'cases':rows}
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.report: Path(args.report).write_text(text+'\n',encoding='utf-8')
    print(text)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())
