"""Post-seal saved-operand responses and finite norm/product decompositions."""
import gc
import json
from pathlib import Path
from encoder_atoms import np,save,sha
from paired_fields import load,retain,metrics
from first_consumer import STAGES,ln_terms,product_terms,silu64,energy_terms,response_account


def execute(root,primary,out):
    from source_response import mask
    root=Path(root);out=Path(out);selected=mask('bottom-right');scopes=[('all',np.ones(1024,bool)),('source-nonlocal',selected)];stage_fields={};stage_accounts=[];factor_fields={};factor_accounts=[];query_records=0;factor_records=0
    def get(stage,r,h):
        if stage in ['projection','gatedCarrier']:return load(primary['shared'][h][stage])
        cell=primary['cells'][r+h]
        if stage in ['gateOperand','upOperand']:
            x=load(cell['fused']);return x[...,:9216].copy() if stage=='gateOperand' else x[...,9216:].copy()
        return load(cell[stage])
    with (out/'queries.jsonl').open('x') as stream:
        for stage in STAGES:
            a,b,c,d=[get(stage,r,h) for r,h in [('R0','A'),('R0','B'),('R1','A'),('R1','B')]];d0=b-a;d1=d-c;interaction=d1-d0
            stage_fields[stage]={n:retain(root/'arrays/responses'/stage,n,x) for n,x in [('responseR0',d0),('responseR1',d1),('interaction',interaction)]}
            for scope,sel in scopes:stage_accounts.append({'stage':stage,'scope':scope,'queries':int(sel.sum()),**response_account(d0[0,sel],d1[0,sel])})
            for index in range(1024):stream.write(json.dumps({'stage':stage,'imageQueryIndex':index,'sourceNonlocal':bool(selected[index]),**response_account(d0[0,index],d1[0,index])},allow_nan=False)+'\n');query_records+=1
            del a,b,c,d,d0,d1,interaction;gc.collect()
    plus=load(primary['constants']['onePlusScale']);wa,wb=[get('gatedCarrier','R0',h) for h in ['A','B']]
    with (out/'factors.jsonl').open('x') as stream:
        for r in ['R0','R1']:
            ua,ub=[get('residualCommit',r,h) for h in ['A','B']];na,nb=[get('normalization',r,h) for h in ['A','B']];sa,sb=[get('normScaled',r,h) for h in ['A','B']];za,zb=[get('mlpInput',r,h) for h in ['A','B']]
            terms=ln_terms(ua,ub);native=nb-na;policy=native-terms['reference'];addition=(ub-ua)-(wb-wa);scale_policy=(sb-sa)-native*plus;shift_policy=(zb-za)-(sb-sa)
            values={**terms,'nativePolicyDifference':policy,'residualAdditionRounding':addition,'scalePolicyDifference':scale_policy,'shiftAdditionRounding':shift_policy};factor_fields[r+'.normalization']={n:retain(root/'arrays/factors'/r/'normalization',n,x) for n,x in values.items()}
            def norm_account(sel):
                sl=lambda x:x[0,sel]
                return {**energy_terms(sl(terms['direct']),sl(terms['rescale']),sl(terms['reference'])),'nativePolicy':metrics(sl(native),sl(terms['reference'])),'residualAdditionRoundingRms':float(np.sqrt(np.mean(sl(addition)**2))),
                    'scalePolicyRms':float(np.sqrt(np.mean(sl(scale_policy)**2))),'shiftAdditionRoundingRms':float(np.sqrt(np.mean(sl(shift_policy)**2))),'maxIdentityClosureAbs':float(np.max(np.abs(sl(terms['closure']))))}
            for scope,sel in scopes:factor_accounts.append({'receiver':r,'kind':'normalization','scope':scope,'queries':int(sel.sum()),**norm_account(sel)})
            for index in range(1024):stream.write(json.dumps({'receiver':r,'kind':'normalization','imageQueryIndex':index,'sourceNonlocal':bool(selected[index]),**norm_account(index)},allow_nan=False)+'\n');factor_records+=1
            del ua,ub,na,nb,sa,sb,za,zb,terms,native,policy,addition,scale_policy,shift_policy,values;gc.collect()
            ga,gb=[get('gateOperand',r,h) for h in ['A','B']];va,vb=[get('upOperand',r,h) for h in ['A','B']];fa,fb=[get('activation',r,h) for h in ['A','B']];pa,pb=[get('product',r,h) for h in ['A','B']]
            terms=product_terms(fa,fb,va,vb);native=pb-pa;product_policy=native-terms['reference'];reference_activation=silu64(gb)-silu64(ga);native_activation=fb-fa;activation_policy=native_activation-reference_activation
            values={**terms,'nativeProductPolicyDifference':product_policy,'referenceActivationDelta':reference_activation,'nativeActivationPolicyDifference':activation_policy};factor_fields[r+'.product']={n:retain(root/'arrays/factors'/r/'product',n,x) for n,x in values.items()}
            def product_account(sel):
                sl=lambda x:x[0,sel]
                return {**energy_terms(sl(terms['gate']),sl(terms['up']),sl(terms['reference'])),'nativeProductPolicy':metrics(sl(native),sl(terms['reference'])),'nativeActivationPolicy':metrics(sl(native_activation),sl(reference_activation)),'maxIdentityClosureAbs':float(np.max(np.abs(sl(terms['closure']))))}
            for scope,sel in scopes:factor_accounts.append({'receiver':r,'kind':'product','scope':scope,'queries':int(sel.sum()),**product_account(sel)})
            for index in range(1024):stream.write(json.dumps({'receiver':r,'kind':'product','imageQueryIndex':index,'sourceNonlocal':bool(selected[index]),**product_account(index)},allow_nan=False)+'\n');factor_records+=1
            del ga,gb,va,vb,fa,fb,pa,pb,terms,native,product_policy,reference_activation,native_activation,activation_policy,values;gc.collect()
    save(out/'response-fields.json',stage_fields);save(out/'factor-fields.json',factor_fields)
    return {'stageAccounts':stage_accounts,'factorAccounts':factor_accounts,'queryRecords':query_records,'factorRecords':factor_records,'additionalFP64SiLUScalarEvaluations':4*1024*9216,'additionalFP64LayerNormScalarEvaluations':4*1024*3072}
