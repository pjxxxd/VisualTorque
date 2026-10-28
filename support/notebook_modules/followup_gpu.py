"""Final-token GPU helpers embedded after the proven adapter/scoring definitions."""

def final_forward(example, capture_layers=(), operations=None):
    operations=operations or {}
    prepared=prepare_examples([example]);mask=final_mask(prepared)
    captured={};audits={};handles=[]
    for layer in sorted(set(capture_layers)|set(operations)):
        def hook(_module,_inputs,output,layer=layer):
            hidden=output_hidden_tensor(output);m=mask.to(hidden.device)
            assert int(m.sum())==1 and hidden.dtype==torch.bfloat16
            pre=hidden[0,m];post=pre;op=operations.get(layer)
            if op:
                cast=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=pre.dtype).float().numpy()
                h=pre.detach().float().cpu().numpy()
                if op['action']=='add':
                    delta=torch.as_tensor(op['payload'],device=pre.device,dtype=pre.dtype).reshape(pre.shape)
                    post=pre+delta
                    audits[layer]=dict(action='add',requested_norm=float(delta.float().norm()))
                elif op['action']=='matched_add':
                    target=torch.as_tensor(op['reference'],device=pre.device,dtype=pre.dtype).reshape(pre.shape)
                    desired=float(((pre+target).float()-pre.float()).norm())
                    r=random_unit(h.shape,op['key'])
                    value,audit=native_norm_edit(h,r,desired,cast)
                    post=torch.as_tensor(value,device=pre.device,dtype=pre.dtype)
                    audits[layer]=dict(action='matched_add',**audit)
                elif op['action'] in ('clamp','random_coordinate','equal_remove'):
                    base=torch.as_tensor(op['untreated'],device=pre.device,dtype=pre.dtype).reshape(pre.shape)
                    axis=torch.as_tensor(op['axis'],device=pre.device,dtype=torch.float32).reshape(pre.shape)
                    axis=axis/axis.norm()
                    if op['action']=='random_coordinate':
                        axis=torch.as_tensor(random_unit(h.shape,op['key']),device=pre.device,dtype=torch.float32)
                    coefficient=((pre.float()-base.float())*axis).sum(-1,keepdim=True)
                    reference=(pre.float()-coefficient*axis).to(pre.dtype)
                    target_norm=float((reference.float()-pre.float()).norm())
                    if op['action']=='equal_remove':
                        sign=1. if float(coefficient.item())>=0 else -1.
                        value,audit=native_norm_edit(h,-sign*random_unit(h.shape,op['key']),target_norm,cast)
                        post=torch.as_tensor(value,device=pre.device,dtype=pre.dtype)
                    else:
                        post=reference;audit=dict(target_norm=target_norm,matched=True,relative_error=0.)
                    audits[layer]=dict(action=op['action'],coefficient=float(coefficient.item()),**audit)
                else:raise ValueError(op['action'])
                audits[layer]['actual_norm']=float((post.float()-pre.float()).norm())
                audits[layer]['hidden_norm']=float(pre.float().norm())
                modified=hidden.clone();modified[0,m]=post
            captured[layer]=post.detach().float().cpu().numpy().copy()
            return replace_output_hidden(output,modified) if op else None
        handles.append(DECODER_LAYERS[layer].register_forward_hook(hook))
    try:
        with torch.inference_mode():out=adapter.model(**prepared['inputs'],use_cache=False)
    finally:
        for handle in handles:handle.remove()
    record=score_records(prepared,out.logits)[0]
    record.update(example_id=example.example_id,scene_id=str(example.scene_id),variant=example.variant,
        split=example.split,net_torque=int(example.net_torque),correct=record['predicted_semantic']==str(example.gold_semantic),
        write_audits={str(k):v for k,v in audits.items()})
    assert set(captured)==set(capture_layers)|set(operations)
    return record,captured


def example_stamp(example):
    return dict(example_id=example.example_id,prompt=example.prompt,image=file_hash(example.image_path),
        labels=list(example.labels),semantics=list(example.option_semantics),gold=str(example.gold_semantic),
        torque=int(example.net_torque),split=example.split)


def current_runtime_identity():
    import transformers
    return dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
        dtype=MODEL_DTYPE,batch_size=1,checkpoint=MODEL_REVISION,layer_path=LAYER_PATH,
        implementation=IMPLEMENTATION_SHA256,
        toolkit={str(p.relative_to(TOOLKIT_ROOT)):file_hash(p) for p in sorted((TOOLKIT_ROOT/'src').rglob('*.py'))})
