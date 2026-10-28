# %% Configure one CPU run for both completed models
RUN_ID='main'
SOURCE_RUN_ID='main'
AUTO_DOWNLOAD=True
BASE=PROJECT_ROOT/'results/cross_model_independent_confirmation'
OUTPUT_ROOT=BASE/'error_structure'/RUN_ID
packet=DiagnosticPacket(OUTPUT_ROOT,dict(schema='error-structure-1.0',implementation=IMPLEMENTATION_SHA256,
    source_run=SOURCE_RUN_ID,models=['qwen27b','gemma12b'],new_scientific_confirmation=False,
    bootstrap='5000 family-cluster draws; seed-average numerator/denominator; descriptive intervals only',
    changes_to_existing_inputs=False,gpu_forwards=0))
roots={'qwen27b':BASE/'qwen_three_path'/SOURCE_RUN_ID/'Qwen__Qwen3.5-27B__373612cb',
       'gemma12b':BASE/'gemma_l31_candidate'/SOURCE_RUN_ID/'google__gemma-4-12B-it'}
selected={'qwen27b':['zero','u','u_random','a','a_random','b','bypass'], 'gemma12b':['zero','targeted','random']}
bundles={};coverage=[];sections=[]
print('CPU 一次读取 Qwen-27B 和 Gemma，无需切换模型。输出：',OUTPUT_ROOT)

# %% Read actual answers and independently check the physical labels
for model,root in tqdm(roots.items(),desc='Reading saved answers'):
    try:
        bundle=load_science_result(root,model,packet.inputs)
        if bundle['status']=='validated_complete':
            answers=behavior_rows(bundle)
            packet.csv(model+'_all_saved_answer_details.csv',answers)
            packet.json(model+'_physical_families.json',bundle['families'])
            packet.json(model+'_original_completion.json',bundle['completion'])
            subset=answers[answers.condition.isin(selected[model])].copy()
            assert subset.valid.all(),'Selected repair conditions include an invalid technical control.'
            bundles[model]=(bundle,subset)
        coverage.append(dict(model=model,status=bundle['status'],issues='; '.join(bundle['issues'])))
    except (AssertionError,ValueError,KeyError,TypeError,OSError) as error:
        coverage.append(dict(model=model,status='input_integrity_error',issues=type(error).__name__+': '+str(error)))
packet.csv('coverage.csv',pd.DataFrame(coverage))
diagnostic_show('输入覆盖情况',pd.DataFrame(coverage))

# %% Count errors, heuristic choices and repairs with explicit denominators
for model,(bundle,answers) in bundles.items():
    rates,confusion=error_rate_tables(answers)
    packet.csv(model+'_error_rates_all_groups.csv',rates)
    packet.csv(model+'_answer_confusion_counts.csv',confusion)
    native=rates[(rates.condition=='zero') & rates['group'].isin(['stratum','gold']) &
        rates.metric.isin(['accuracy','weight_choice_on_directional_conflicts','weight_choice_when_weight_disagrees'])]
    shown=native[['group','value','metric','numerator_seed_average','denominator_seed_average','pooled_fraction','lower','upper']]
    diagnostic_show(model+'：原模型准确率与重量预测一致率（分母明确列出）',shown,40)
    repair=rates[(rates['group']=='gold') & rates.metric.isin(['repair_of_original_errors','harm_of_original_correct'])]
    diagnostic_show(model+'：左右各自的修复率与新增错误率',repair[['condition','value','metric','numerator_seed_average','denominator_seed_average','pooled_fraction']],35)
    diagnostic_show(model+'：原模型错误实际选向哪里',confusion[confusion.condition=='zero'],30)
    sections.append('## '+model+'：错误类型与左右表现\n\n'+csv_markdown(shown))

# %% Compare mirrored scenes and option-letter swaps within each family
for model,(bundle,answers) in bundles.items():
    mirror=paired_behavior(answers,'mirror')
    mapping=paired_behavior(answers,'mapping')
    pair_table=pd.concat([mirror,mapping],ignore_index=True)
    summary=pair_summaries(pair_table)
    changes=paired_consistency_changes(pair_table)
    packet.csv(model+'_mirror_pair_details.csv',mirror)
    packet.csv(model+'_mapping_pair_details.csv',mapping)
    packet.csv(model+'_pair_consistency_summary.csv',summary)
    packet.csv(model+'_paired_consistency_change_vs_zero.csv',changes)
    important=summary[summary.metric.isin(['semantic_consistent','both_correct','same_letter_different_semantics'])]
    diagnostic_show(model+'：镜像应交换左右；换选项字母应保持物理答案',important[['pairing','condition','metric','numerator_seed_average','denominator_seed_average','pooled_fraction','lower','upper']],40)
    sections.append('## '+model+'：镜像和答案映射\n\n'+csv_markdown(important))
    diagnostic_show(model+'：相对未修改答案的一致性变化（同一家族内配对）',changes,25)

# %% Save the complete CPU evidence package
summary=dict(status='complete' if len(bundles)==2 else 'complete_with_missing_or_invalid_inputs',
    models_completed=list(bundles),coverage=coverage,gpu_forward_count=0,
    original_scientific_decisions_unchanged=True,new_confirmatory_tests=0)
report='\n\n'.join(['# 已保存答案的错误结构分析',
    '只分析已有 Qwen/Gemma 独立场景记录，不新增模型前向，不改原确认状态。',
    '## 覆盖情况',csv_markdown(pd.DataFrame(coverage)),*sections,
    '## 解释范围',
    '重量预测为两侧总重量之差；距离预测忽略重量，比较各侧所有堆到支点的距离之和。与启发式预测一致不自动证明内部采用了该算法。',
    'directional conflicts 排除重量相同的场景；另一个 weight disagrees 指标包含等重时预测平衡的情况。错误类别允许两种启发式共同预测同一错误答案，并单列为 both_heuristics。',
    '所有比率都给出分子和分母。随机对照先按种子平均；缺少符合条件的分母时显示缺失，不能写成 0%。区间按整个物理家族重采样，不把镜像、画风或映射当成独立样本。',
    '镜像一致包含“平衡→平衡”，不等于答对；both_correct 单独报告。换字母后同一字母对应不同物理答案，才计入 same_letter_different_semantics。',
    '这里仅有原先固定的两种答案映射，不能外推到所有提示或六种映射；两模型不是相同家族，不作配对性能排名。',
    '本轮均为事后描述，不能自动证明方向偏差的原因、提高原确认状态或宣称新的物理机制。'])+'\n'
packet.finish(summary,report,'VisualTorque_Error_Structure_Output.zip',AUTO_DOWNLOAD)
