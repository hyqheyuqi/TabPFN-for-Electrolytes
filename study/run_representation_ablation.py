"""Step 4 CLI: local physical-representation ablation; no API requests."""
from pathlib import Path
import argparse
import contextlib
import sys
from analysis.representation_ablation import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['pilot','full','plots'])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--reuse-local-results',type=Path,help='Reuse unchanged routes from existing local raw results')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=args.output_dir or Path(__file__).resolve().parent/'results/representation_ablation'
    if args.mode!='plots' and output.exists() and any(output.iterdir()):
        parser.error('Choose an empty output directory; plots rebuilds completed results')
    (results/'logs').mkdir(parents=True,exist_ok=True)
    class Tee:
        def __init__(self,stream,log):self.stream,self.log=stream,log
        def write(self,value):self.stream.write(value);self.log.write(value);return len(value)
        def flush(self):self.stream.flush();self.log.flush()
    with (results/'logs/execution.log').open('a',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            run(root,output,args.mode,args.reuse_local_results)
