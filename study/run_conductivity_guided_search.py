"""Step 3 CLI: audited property cards and four-acquisition offline comparison."""
from pathlib import Path
import argparse
import contextlib
import sys
from analysis.conductivity_guided_search import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','pilot','full','plots'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output_dir or Path(__file__).resolve().parent/'results/conductivity_guided_search'
    if args.mode not in ('audit','plots') and output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output exists; pass --resume or choose an empty --output-dir')
    class Tee:
        def __init__(self,stream,log):self.stream,self.log=stream,log
        def write(self,value):self.stream.write(value);self.log.write(value);return len(value)
        def flush(self):self.stream.flush();self.log.flush()
    (results/'logs').mkdir(parents=True,exist_ok=True)
    with (results/'logs/execution.log').open('a',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            run(root,output,args.mode,args.resume)
