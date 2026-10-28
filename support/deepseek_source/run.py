"""CLI used by the three Colab notebooks. A failed stage never writes a completion marker."""
import argparse
import json
from pathlib import Path
import traceback
from core import base_namespace, preflight, export

STAGES=['preflight','calibration','validation','formats','readout-fit','readout-validation',
        'repair-development','repair-confirmation','summary','export']


def summary(n):
    root=n['ROOT']; rows=[]
    markers={
        '4.2–4.3 ABC':root/'prediction/models/deepseek16b/validation/completion.json',
        '4.3 XYZ/digits':root/'prediction/models/deepseek16b/formats/completion.json',
        '4.4 readout/correction':root/'readout/completion.json',
        '4.5 repair/removal':root/'repair/confirmation/completion.json'}
    for name,path in markers.items():
        rows.append(dict(section=name,complete=path.is_file(),details=n['read_json'](path) if path.is_file() else None))
    n['save_json'](root/'summary.json',rows)
    lines=['# DeepSeek-VL2-Small 实验状态','',
        '完成标记表示采集与分析完成，不表示论文结论已复现。缺少标记的阶段不能当作完整实验。','']
    for r in rows:lines.append('- '+r['section']+'：'+('已完成' if r['complete'] else '未完成'))
    candidates=[
        ('4.2 映射收益','prediction/models/deepseek16b/validation/analysis/mapping_contrasts.csv'),
        ('4.3 预测比较','prediction/models/deepseek16b/validation/analysis/primary_comparisons.csv'),
        ('4.3 新符号','prediction/models/deepseek16b/formats/analysis/primary_comparisons.csv'),
        ('4.4 数值修准','readout/correction_summary.csv'),
        ('4.5 修复与依赖','repair/confirmation/primary_tests.csv')]
    for label,relative in candidates:
        p=root/relative
        if p.is_file():
            lines.extend(['','## '+label,'','```text',n['pd'].read_csv(p).to_string(index=False),'```'])
    lines.extend(['','解释顺序：先检查基线、事件数和技术失败，再查看区间与校正后的比较。',
        '4.2 的不显著结果不能写成映射不影响；4.4 只测试此读出器与修准操作；',
        '4.5 只有初始修复成立时才能解释依赖；三条局部关系不等同完整计算链。'])
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:8]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True);p.add_argument('--run-id',default='blackwell_cu128')
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--source-support',default=None,help='Optional original support.json; otherwise detect the canonical original path.')
    args=p.parse_args();n=base_namespace(args.project,args.run_id)
    try:
        if args.stage=='preflight':preflight(n)
        elif args.stage in ('calibration','validation'):
            import prediction;prediction.run(n,args.stage,args.source_support)
        elif args.stage=='formats':
            import prediction;prediction.formats(n,args.source_support)
        elif args.stage=='readout-fit':
            import readout;readout.fit(n)
        elif args.stage=='readout-validation':
            import readout;readout.validate(n)
        elif args.stage=='repair-development':
            import repair;repair.development(n)
        elif args.stage=='repair-confirmation':
            import repair;repair.confirmation(n)
        elif args.stage=='summary':summary(n)
        elif args.stage=='export':summary(n);export(n)
    except Exception:
        n['save_json'](n['ROOT']/('failure_'+args.stage+'.json'),dict(stage=args.stage,status='failed_not_complete',
            traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
