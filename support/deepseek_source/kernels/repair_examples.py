def repair_prompt(task,semantics):
    question={'outcome':'Which way will the balance scale tip?','weight_more':'Which side has more total weight?',
              'weight_less':'Which side has less total weight?'}[task]
    return '\n\n'.join(['Answer this question using only the supplied balance-scale scene.',RULE_TEXT,'Question: '+question,
        'Options:\n'+'\n'.join(f'{l}. {SEMANTIC_TEXT[s]}' for l,s in zip(LABELS,semantics)),
        'Do not explain your reasoning. Reply with exactly one option letter and nothing else. Valid replies are: A, B, C.'])

def new_example(sid,path,features,semantics,variant,split,task='outcome'):
    val=features['net_torque'] if task=='outcome' else features['weight_score']*(1 if task=='weight_more' else -1)
    gold='left' if val>0 else 'right' if val<0 else 'balanced'
    return BehaviorExample(example_id=f'followup:{sid}:{task}:{variant}',question_id=f'followup_{sid}_{task}_{variant}',
        scene_id=sid,condition=PromptCondition.IMAGE_RULE,question_kind=task,variant=variant,split=split,
        problem_type='fixed_physics_repair',outcome=gold,net_torque=int(features['net_torque']),
        prompt=repair_prompt(task,semantics),image_path=Path(path),labels=LABELS,
        option_text=tuple(SEMANTIC_TEXT[s] for s in semantics),option_semantics=tuple(semantics),
        gold_label=LABELS[semantics.index(gold)],gold_semantic=gold)

def two_mappings(scene, family_id):
    # Rotation is family-fixed; mirror and render variants stay in the same family.
    offset=int(digest(['repair_mapping',family_id])[:8],16)%3
    base=['left','balanced','right'];base=base[offset:]+base[:offset]
    reverse=[{'left':'right','right':'left','balanced':'balanced'}[x] for x in base]
    return {'forward':tuple(base),'reverse':tuple(reverse)}

def code_key(wrong_label,correct_label):return str(wrong_label)+'_'+str(correct_label)
