"""Bounded, TEST-only hypothesis experiments for the local image sandbox.

No surrogate predictions: every experimental arm is evaluated by the real engine.
Images/repeats stay paired; uncertainty is conditional on the selected cohort.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import io
import importlib.metadata
import secrets
import threading

import numpy as np
from PIL import Image
from scipy import ndimage

LIMITS = {'none': (0, 1), 'rotation': (-45, 45), 'translation': (-6, 6),
          'brightness': (0, 2), 'saturation': (0, 2), 'contrast': (0, 2), 'blur': (0, 3),
          'noise': (0, .3), 'red_scale': (0, 2), 'green_scale': (0, 2), 'blue_scale': (0, 2),
          'occlusion': (1, 12), 'pc1': (-2, 2), 'pc2': (-2, 2), 'pc_plane': (0, 2)}
FEATURE_NAMES = ['Mean brightness', 'Mean saturation', 'Mean red', 'Mean green', 'Mean blue', 'Luminance contrast (std)', 'Boundary gradient fraction']
PRESETS = [
    dict(id='appearance', title='RGB/brightness/border-texture associations', question='Do matched errors differ in color or border texture, and which edits change the predictions?',
         test='Inspect all seven original notebook features on matched error/control pairs; compare channel scaling, brightness and blur while tracking their coupled feature changes.',
         against='An association that disappears under tight matching, or changes with several correlated features, does not identify a causal feature.',
         kind='blue_scale', level=.9, levels=[.9, .95, 1, 1.05, 1.1], comparison='channels', cohort='matched', group='all'),
    dict(id='shared', title='Shared versus model-specific blind spots', question='Do the same images and classes defeat all three models?',
         test='Compare shared-HC, exclusive-error and IG/neutrophil image groups; inspect class substitutions and all eight probabilities.',
         against='Different destination classes or correct predictions in other models argue against a shared blind spot.',
         kind='none', level=1, levels=[1], comparison='sweep', cohort='filtered', group='shared_hc'),
    dict(id='confidence', title='Confidence versus correctness', question='Does confidence remain trustworthy under intervention?',
         test='Compare confidence, HC-error rate, Brier score and descriptive ECE before/after; inspect tightly matched error/control pairs.',
         against='High confidence on wrong predictions contradicts confidence-as-correctness; a small selected cohort cannot establish global calibration.',
         kind='brightness', level=1.05, levels=[.95, 1, 1.02, 1.05, 1.1], comparison='sweep', cohort='matched', group='all'),
    dict(id='saturation', title='Saturation → erythroblast attractor', question='Does desaturation drive both errors and correct controls toward erythroblast?',
         test='Sweep saturation on one-to-one same-class, similar-confidence pairs; inspect destination classes and paired error-minus-control effects.',
         against='No destination-class concentration, or an effect restricted to one cohort/class, weakens the broad attractor explanation.',
         kind='saturation', level=.5, levels=[.5, .75, 1, 1.25, 1.5], comparison='sweep', cohort='matched', group='all'),
    dict(id='roi', title='Stained object versus background color', question='Which approximate spatial region carries the color sensitivity?',
         test='Compare whole/foreground/complement at identical color settings; inspect/edit masks, then compare core versus expanded masks.',
         against='A contrast that reverses with mask choice or perturbation dose is not convincing anatomical localization.',
         kind='saturation', level=.5, levels=[.5, .75, 1, 1.25, 1.5], comparison='scopes', cohort='current', group='all'),
    dict(id='spatial', title='Central versus peripheral occlusion', question='Is central sensitivity actually greater in matched HC errors?',
         test='Compare equal-area center/corner gray patches on tight pairs; read the paired support-loss contrast, not population sensitivity alone.',
         against='Error-minus-control intervals covering zero weaken an error-specific spatial explanation, as in the tighter notebook analysis.',
         kind='occlusion', level=4, levels=[2, 3, 4, 6, 8], comparison='positions', cohort='matched', group='all'),
    dict(id='flips', title='Small-change prediction flips', question='Can a tiny change flip an initially confident correct decision?',
         test='Load a recorded mild-flip example or sweep around the identity; inspect exact images and all model/class probabilities.',
         against='Stable decisions in a local sweep weaken instability for that image, not for the entire TEST population.',
         kind='translation', level=1/3, levels=[-1, -1/3, 0, 1/3, 1], comparison='sweep', cohort='current', group='all'),
    dict(id='resize', title='Noise gap versus resizing/interpolation', question='How much of the apparent noise advantage depends on resizing?',
         test='Use identical noisy28 inputs under native/nearest/bilinear/bicubic; compare both absolute accuracy and change from each mode’s own clean baseline.',
         against='A changed clean baseline makes architecture-only explanations invalid even when the noise-loss gap narrows; ResNet50 stays native28.',
         kind='noise', level=.1, levels=[0, .01, .03, .05, .1], comparison='resize', cohort='balanced', group='all'),
    dict(id='ig', title='IG ↔ neutrophil class competition', question='Which edits move probability between immature granulocytes and neutrophils?',
         test='Filter to these true classes, vary appearance, inspect directed class substitutions and the full probability vectors.',
         against='Moves into other classes or strong class-dependent asymmetry weaken a simple two-class explanation.',
         kind='brightness', level=1.02, levels=[.95, 1, 1.02, 1.05, 1.1], comparison='sweep', cohort='filtered', group='ig_neut_all'),
    dict(id='pca', title='Advanced: actual-model PCA-direction probes', question='Do local changes along leading pixel-PC directions alter actual predictions?',
         test='Move along PC1/PC2 in standard-deviation units while preserving the image residual; inspect clipping and actual class probabilities.',
         against='A stable or clipping-dependent local probe cannot validate a global 2D boundary; no surrogate is used here.',
         kind='pc1', level=.5, levels=[-1, -.5, 0, .5, 1], comparison='sweep', cohort='current', group='all'),
]


class BusyError(RuntimeError):
    pass


class Workbench:
    def __init__(self, root, images, labels, baseline, classes, engine, transform, png):
        self.root, self.images, self.y, self.p0, self.classes = root, images, labels, baseline, classes
        self.engine, self.transform, self.png = engine, transform, png
        self.pred, self.conf = baseline.argmax(-1), baseline.max(-1)
        self.wrong = self.pred != labels
        self.hc = self.wrong & (self.conf >= .9)
        self.batch_lock = threading.Lock()
        with np.load(root/'predictions/targeted_followup.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['baseline_probabilities'],baseline)
            np.testing.assert_array_equal(z['labels'],labels)
            self.pairs = [z[f'matching_{name}_pairs'] for name in ['resnet18', 'resnet50', 'vit']]
            self.examples = [dict(id=int(i), kind=str(k), level=float(v)) for i,k,v in zip(z['examples_ids'],z['examples_kinds'],z['examples_levels'])]
        with np.load(root/'archived/predictions/test_pca.npz', allow_pickle=False) as z:
            assert str(z['input_sha256']) == hashlib.sha256(images.tobytes()).hexdigest()
            self.components, self.pc_std = z['components'], np.sqrt(z['explained_variance'])
            self.pc_variance = z['explained_variance_ratio'].tolist()
        with np.load(root/'predictions/stability.npz',allow_pickle=False) as z:
            self.notebook_levels={key[7:]:z[key].tolist() for key in z.files if key.startswith('levels_')}
        with np.load(root/'predictions/high_confidence_error_analysis.npz',allow_pickle=False) as z:
            self.notebook_levels['saturation']=z['saturation_levels'].tolist()
            np.testing.assert_allclose(self.features(images),z['features'],atol=1e-12,rtol=1e-10)
        self.input_sha = hashlib.sha256(images.tobytes()).hexdigest()
        self.runtime={name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','torchvision','transformers','Pillow']}
        app_root=Path(__file__).resolve().parent
        self.implementation_sha={name:hashlib.sha256((app_root/name).read_bytes()).hexdigest() for name in ['image_sandbox.py','hypothesis_workbench.py','index.html']}

    @staticmethod
    def features(images):
        rgb=images.astype(float)/255;luminance=rgb@np.array([.299,.587,.114])
        maximum,minimum=rgb.max(-1),rgb.min(-1)
        saturation=np.divide(maximum-minimum,maximum,out=np.zeros_like(maximum),where=maximum>0)
        gy,gx=np.gradient(luminance,axis=(1,2));energy=np.hypot(gx,gy)
        border=np.ones((28,28),bool);border[4:-4,4:-4]=False
        boundary=energy[:,border].sum(-1)/np.maximum(energy.sum((1,2)),1e-12)
        return np.column_stack([luminance.mean((1,2)),saturation.mean((1,2)),rgb.mean((1,2)),luminance.std((1,2)),boundary])

    def index(self, value):
        i = int(value)
        if not 0 <= i < len(self.y): raise ValueError('Invalid TEST ID')
        return i

    @staticmethod
    def mode(value):
        if value not in ['native', 'nearest', 'bilinear', 'bicubic']: raise ValueError('Unknown resize mode')
        return value

    @staticmethod
    def decode(url, mask=False):
        im = Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[-1], validate=True)))
        if im.size != (28,28): raise ValueError('Images/masks must be 28×28')
        return np.asarray(im.convert('L' if mask else 'RGB')).copy()

    def decode_input(self, request, mask=False):
        raw_key='mask_pixels' if mask else 'pixels'
        if raw_key in request:
            raw=base64.b64decode(request[raw_key],validate=True)
            shape=(28,28) if mask else (28,28,3)
            if len(raw)!=int(np.prod(shape)):raise ValueError('Invalid raw pixel buffer length')
            return np.frombuffer(raw,dtype=np.uint8).reshape(shape).copy()
        return self.decode(request['mask' if mask else 'image'],mask)

    def mask(self, image, name='core', manual=None):
        if name == 'manual':
            if manual is None: raise ValueError('Paint a manual mask first')
            binary = manual.astype(float)/255
        else:
            if name not in ['core','expanded']: raise ValueError('Unknown mask')
            rgb=image.astype(float);border=np.ones((28,28),bool);border[2:-2,2:-2]=False
            distance=np.linalg.norm(rgb-np.median(rgb[border],axis=0),axis=-1)
            binary=ndimage.binary_closing(distance>max(15.,float(np.quantile(distance[border],.90))+5.))
            components,n=ndimage.label(binary)
            if n:
                yy,xx=np.indices((28,28));scores=[]
                for k in range(1,n+1):
                    component=components==k
                    scores.append(component.sum()*np.exp(-((yy[component].mean()-13.5)**2+(xx[component].mean()-13.5)**2)/98))
                binary=ndimage.binary_fill_holes(components==int(np.argmax(scores))+1)
            if name=='expanded':binary=ndimage.binary_dilation(binary,iterations=2)
        return np.asarray(binary,dtype=float)

    def pool(self, request):
        group=request.get('group','all');m=int(request.get('model',0))
        if not 0<=m<3:raise ValueError('Unknown matching/filter model')
        groups={'all':np.ones(len(self.y),bool),'errors':self.hc.any(0), 'shared':self.wrong.all(0),
                'shared_hc':self.hc.all(0),'exclusive':self.hc[m]&(self.wrong.sum(0)==1),
                'correct':(~self.wrong).all(0), 'ig_neut_all':np.isin(self.y,[3,6]),
                'ig_neut':(self.hc&(((self.y==3)&(self.pred==6))|((self.y==6)&(self.pred==3)))).any(0)}
        if group not in groups:raise ValueError('Unknown image group')
        eligible=groups[group].copy()
        for key,values in [('true_class',self.y),('predicted_class',self.pred[m])]:
            cls=int(request.get(key,-1))
            if cls not in range(-1,8):raise ValueError('Invalid class filter')
            if cls>=0:eligible &= values==cls
        ids=np.flatnonzero(eligible)
        if not len(ids):raise ValueError('No TEST images match those filters')
        return ids

    def image(self, request):
        i=self.index(request['id']) if 'id' in request else int(secrets.choice(self.pool(request)))
        mask_arrays={name:np.rint(self.mask(self.images[i],name)*255).astype(np.uint8) for name in ['core','expanded']}
        masks={name:self.png(value) for name,value in mask_arrays.items()}
        mask_pixels={name:base64.b64encode(value.tobytes()).decode() for name,value in mask_arrays.items()}
        matches=[]
        for pairs in self.pairs:
            hit=np.flatnonzero((pairs==i).any(1))
            candidates=np.flatnonzero(self.y[pairs[:,0]]==self.y[i])
            j=int(hit[0]) if len(hit) else (int(candidates[0]) if len(candidates) else 0)
            matches.append(pairs[j].tolist())
        return dict(id=i,label=int(self.y[i]),classes=self.classes,image=self.png(self.images[i]),
                    pixels=base64.b64encode(self.images[i].tobytes()).decode(),mask_pixels=mask_pixels,
                    baseline=self.p0[:,i].tolist(),masks=masks,matches=matches,features=self.features(self.images[i:i+1])[0].tolist(),
                    shared_error=bool(self.wrong[:,i].all()),shared_hc=bool(self.hc[:,i].all()))

    def catalog(self):
        calibration=[]
        for m in range(3):
            correct=~self.wrong[m];bins=[];ece=0.
            membership=np.minimum((self.conf[m]*10).astype(int),9)
            for b in range(10):
                hit=membership==b
                if hit.any():
                    accuracy=float(correct[hit].mean());confidence=float(self.conf[m,hit].mean())
                    ece+=hit.mean()*abs(accuracy-confidence)
                    bins.append(dict(bin=b,count=int(hit.sum()),accuracy=accuracy,confidence=confidence))
            calibration.append(dict(accuracy=float(correct.mean()),confidence=float(self.conf[m].mean()),ece=ece,hc_errors=int(self.hc[m].sum()),bins=bins))
        return dict(presets=PRESETS,examples=self.examples,classes=self.classes,limits=LIMITS,
                    native_calibration=calibration,pc_variance=self.pc_variance,feature_names=FEATURE_NAMES,notebook_levels=self.notebook_levels,
                    matching_counts=[len(p) for p in self.pairs],max_images=24,max_query_images=768)

    def change(self, image, i, request, *, repeat=0):
        kind=request.get('kind','none');level=float(request.get('level',1))
        if kind not in LIMITS or not np.isfinite(level) or not LIMITS[kind][0]<=level<=LIMITS[kind][1]:
            raise ValueError('Transformation outside permitted range')
        clipping=0.
        if kind=='noise' and request.get('noise_protocol','original')=='fixed_direction':
            key=f"{int(request.get('seed',42))}:{int(i)}:fixed:{repeat}"
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
            changed=np.rint(np.clip(image/255+level*rng.normal(0,1,image.shape),0,1)*255).astype(np.uint8)
        elif kind=='noise' and request.get('noise_protocol','original')!='original':
            raise ValueError('Unknown noise protocol')
        elif kind=='occlusion':
            side=int(round(level));last=28-side;middle=last//2
            positions={'center':(middle,middle),'top_left':(0,0),'top_right':(last,0),'bottom_left':(0,last),'bottom_right':(last,last)}
            position=request.get('position','center')
            if position=='custom':
                x,y=int(request.get('x',0)),int(request.get('y',0))
                if not 0<=x<=last or not 0<=y<=last:raise ValueError('Patch outside image')
            elif position in positions:x,y=positions[position]
            else:raise ValueError('Unknown patch position')
            changed=image.copy();changed[y:y+side,x:x+side]=128
        elif kind in ['pc1','pc2','pc_plane']:
            offsets=np.zeros(2)
            if kind=='pc_plane':
                offsets=np.asarray([request.get('pc1',0),request.get('pc2',0)],dtype=float)
                if not np.isfinite(offsets).all() or (np.abs(offsets)>2).any():raise ValueError('PC offsets must be within ±2 SD')
            else:offsets[int(kind[-1])-1]=level
            raw=image.astype(float)/255+((offsets*self.pc_std)@self.components).reshape(28,28,3)
            clipping=float(((raw<0)|(raw>1)).mean());changed=np.rint(np.clip(raw,0,1)*255).astype(np.uint8)
        else:
            changed=self.transform(image,kind,level,test_id=int(i),seed=int(request.get('seed',42)),repeat=repeat)
        scope=request.get('scope','whole')
        if scope not in ['whole','foreground','background']:raise ValueError('Unknown intervention scope')
        area=1.
        if scope!='whole':
            manual=self.decode_input(request,True) if request.get('mask_type')=='manual' else None
            weight=ndimage.gaussian_filter(self.mask(self.images[i],request.get('mask_type','core'),manual),.7)
            if scope=='background':weight=1-weight
            area=float(weight.mean());changed=np.rint(np.clip(image+(changed.astype(float)-image)*weight[...,None],0,255)).astype(np.uint8)
        rms=float(np.sqrt(np.square((changed.astype(float)-image)/255).mean()))
        return changed,dict(area_fraction=area,rms=rms,clipping_fraction=clipping,changed_pixel_fraction=float(np.any(changed!=image,axis=-1).mean()))

    def infer(self, groups):
        """Batch by preprocessing arm; deduplicate identical inputs, including all R50 resize arms."""
        outputs=[np.empty((3,len(images),8),np.float32) for mode,images in groups]
        queries=0
        with self.engine.lock:
            for m in range(3):
                buckets={};lookups=[]
                for mode,images in groups:
                    mode=None if m==1 or mode=='native' else mode
                    keys=[]
                    for image in images:
                        key=image.tobytes();buckets.setdefault(mode,{})[key]=image;keys.append((mode,key))
                    lookups.append(keys)
                computed={}
                for mode,items in buckets.items():
                    probs,_=self.engine.predict_one(m,np.stack(list(items.values())),interpolation=mode)
                    queries+=len(items)
                    for key,p in zip(items,probs):computed[mode,key]=p
                for target,keys in zip(outputs,lookups):target[m]=np.stack([computed[key] for key in keys])
            self.engine.calls+=1
        return outputs,queries

    def predict(self, request):
        i=self.index(request.get('id',0));image=self.decode_input(request);mode=self.mode(request.get('mode','native'))
        repeat=int(request.get('repeat',0))
        if repeat not in range(3):raise ValueError('Noise repeat must be 0, 1 or 2')
        changed,dose=self.change(image,i,request,repeat=repeat)
        (p,clean),queries=self.infer([(mode,changed[None]),(mode,self.images[i:i+1])])
        return dict(probabilities=p[:,0].tolist(),reference=clean[:,0].tolist(),image=self.png(changed),pixels=base64.b64encode(changed.tobytes()).decode(),dose=dose,features=self.features(changed[None])[0].tolist(),
                    model_image_queries=queries,mode=mode,device=str(self.engine.device))

    @staticmethod
    def draws(labels, seed, paired=False):
        # Pairs stay together; true-class strata retain their original cohort sizes.
        units=labels[::2] if paired else labels
        if len(units)<2:return None
        rng=np.random.default_rng(seed)
        selected=np.column_stack([rng.choice(np.flatnonzero(units==c),size=(500,np.sum(units==c))) for c in np.unique(units)])
        return np.stack([2*selected,2*selected+1],axis=-1).reshape(500,-1) if paired else selected

    @staticmethod
    def stats(values, draws):
        mean=values.mean(-1)
        ci=None if draws is None else np.quantile(values[:,draws].mean(-1),[.025,.975],axis=1).T
        return [dict(mean=float(mean[m]),ci=None if ci is None else ci[m].tolist()) for m in range(3)]

    def feature_contrast(self, features, ids, seed):
        errors,controls=features[::2],features[1::2];delta=errors-controls
        draws=self.draws(self.y[ids][::2],seed);sd=controls.std(0)
        raw_ci=None if draws is None else np.quantile(delta[draws].mean(1),[.025,.975],axis=0)
        effects=[]
        for j in range(7):
            effect=None if sd[j]<=1e-12 else float(delta[:,j].mean()/sd[j]);ci=None
            if draws is not None and effect is not None:
                sampled_sd=controls[draws,j].std(1)
                if (sampled_sd>1e-12).all():ci=np.quantile(delta[draws,j].mean(1)/sampled_sd,[.025,.975]).tolist()
            effects.append(dict(error_mean=float(errors[:,j].mean()),control_mean=float(controls[:,j].mean()),
                raw_difference=dict(mean=float(delta[:,j].mean()),ci=None if raw_ci is None else raw_ci[:,j].tolist()),
                standardized=dict(mean=effect,ci=ci)))
        return effects

    def experiment(self, request):
        if not self.batch_lock.acquire(blocking=False):raise BusyError('An experiment is already running')
        try:return self._experiment(request)
        finally:self.batch_lock.release()

    def _experiment(self, request):
        seed=int(request.get('seed',42));rng=np.random.default_rng(seed);cohort=request.get('cohort','current')
        count=int(request.get('count',24));m=int(request.get('model',0));i=self.index(request.get('id',0))
        if not 1<=count<=24 or not 0<=m<3:raise ValueError('Use 1–24 images and a valid model')
        paired=cohort in ['pair','matched'];roles=[]
        if cohort=='current':ids=np.array([i])
        elif paired:
            pairs=self.pairs[m];cls=int(request.get('true_class',-1))
            if cls not in range(-1,8):raise ValueError('Invalid true class')
            if cohort=='pair':
                hit=np.flatnonzero((pairs==i).any(1));same=np.flatnonzero(self.y[pairs[:,0]]==self.y[i])
                chosen=int(hit[0]) if len(hit) else (int(same[0]) if len(same) else 0);pairs=pairs[[chosen]]
            else:
                if cls>=0:pairs=pairs[self.y[pairs[:,0]]==cls]
                if not len(pairs):raise ValueError('No tight pairs for this class/model')
                pairs=pairs[np.sort(rng.choice(len(pairs),min(len(pairs),max(1,count//2)),replace=False))]
            ids=pairs.ravel();roles=['error','control']*len(pairs)
        elif cohort in ['balanced','filtered']:
            pool=np.arange(len(self.y)) if cohort=='balanced' else self.pool(request)
            # Round-robin stratified sampling handles scarce/absent classes without replacement.
            pools=[list(rng.permutation(pool[self.y[pool]==c])) for c in range(8)];chosen=[]
            while len(chosen)<min(count,len(pool)):
                for pool_c in pools:
                    if pool_c and len(chosen)<count:chosen.append(pool_c.pop())
            ids=np.array(chosen)
        else:raise ValueError('Unknown cohort')
        inputs=self.images[ids].copy()
        if cohort=='current' and ('image' in request or 'pixels' in request):inputs[0]=self.decode_input(request)
        if request.get('mask_type')=='manual' and len(ids)>1:raise ValueError('Manual masks apply only to the current image; use automatic masks for cohorts')
        comparison=request.get('comparison','sweep');mode=self.mode(request.get('mode','native'))
        conditions=[]
        if comparison=='sweep':
            levels=request.get('levels',[request.get('level',1)])
            if not isinstance(levels,list) or not 1<=len(levels)<=11:raise ValueError('Use 1–11 severity levels')
            for level in levels:conditions.append(dict(request,level=float(level),label=f'{float(level):.5g}'))
        elif comparison=='scopes':
            conditions=[dict(request,scope=s,label=s) for s in ['whole','foreground','background']]
        elif comparison=='resize':
            conditions=[dict(request,mode=s,label=s) for s in ['native','nearest','bilinear','bicubic']]
        elif comparison=='positions':
            conditions=[dict(request,kind='occlusion',position=s,label=s) for s in ['center','top_left','top_right','bottom_left','bottom_right']]
        elif comparison=='occlusion_grid':
            side=int(round(float(request.get('level',4))))
            if not 1<=side<=12:raise ValueError('Patch side must be 1–12')
            positions=np.rint(np.linspace(0,28-side,7)).astype(int)
            conditions=[dict(request,kind='occlusion',position='custom',x=int(x),y=int(y),label=f'{x},{y}') for y in positions for x in positions]
        elif comparison=='pca_grid':
            if cohort!='current':raise ValueError('The actual-model PCA plane uses one current image only')
            span=float(request.get('level',1))
            if not np.isfinite(span) or not 0<span<=2:raise ValueError('PCA grid span must be >0 and ≤2 SD')
            positions=np.linspace(-span,span,7)
            conditions=[dict(request,kind='pc_plane',scope='whole',pc1=float(x),pc2=float(y),label=f'{x:.2g},{y:.2g}') for y in positions for x in positions]
        elif comparison=='masks':
            scope=request.get('scope','foreground');scope='foreground' if scope=='whole' else scope
            conditions=[dict(request,scope=scope,mask_type=s,label=s) for s in ['core','expanded']]
        elif comparison=='channels':
            conditions=[dict(request,kind=s+'_scale',label=s) for s in ['red','green','blue']]
        else:raise ValueError('Unknown comparison')
        repeats=int(request.get('noise_repeats',3)) if any(c.get('kind')=='noise' for c in conditions) else 1
        if repeats not in [1,3]:raise ValueError('Use one or three noise repeats')
        if len(conditions)*len(ids)*repeats>768:raise ValueError('Experiment exceeds the 768-query-image limit')
        groups=[];reference_indices={};generated=[]
        for condition in conditions:
            arm=self.mode(condition.get('mode',mode))
            if arm not in reference_indices:reference_indices[arm]=len(groups);groups.append((arm,inputs))
            changed=[];doses=[]
            for repeat in range(repeats):
                row=[];dose=[]
                for image,index in zip(inputs,ids):
                    out,metadata=self.change(image,int(index),condition,repeat=repeat);row.append(out);dose.append(metadata)
                changed.append(row);doses.append(dose)
            changed=np.asarray(changed);group_index=len(groups);groups.append((arm,changed.reshape(-1,28,28,3)))
            generated.append((condition,arm,changed,doses,group_index))
        observed,queries=self.infer(groups);draws=self.draws(self.y[ids],seed,paired)
        variants=[];input_features=self.features(inputs)
        for condition,arm,changed,doses,group_index in generated:
            p=observed[group_index].reshape(3,repeats,len(ids),8);clean=observed[reference_indices[arm]]
            predicted=p.argmax(-1);confidence=p.max(-1);was=clean.argmax(-1);original=clean.max(-1)
            retained=np.take_along_axis(p,was[:,None,:,None],axis=-1)[...,0]
            accuracy=(predicted==self.y[ids]);before=was==self.y[ids]
            onehot=np.eye(8)[self.y[ids]]
            measurements=dict(accuracy=100*accuracy.mean(1),accuracy_change=100*(accuracy.mean(1)-before),
                flips=100*(predicted!=was[:,None]).mean(1),hc_errors=100*((~accuracy)&(confidence>=.9)).mean(1),
                confidence=100*confidence.mean(1),support_loss=100*(original-retained.mean(1)),
                true_class_probability=100*np.take_along_axis(p,self.y[ids][None,None,:,None],axis=-1)[...,0].mean(1),
                entropy=(-(p*np.log(np.maximum(p,1e-12))).sum(-1)/np.log(8)).mean(1),brier=np.square(p-onehot).sum(-1).mean(1))
            metrics={key:self.stats(value,draws) for key,value in measurements.items()}
            metrics['clean_accuracy']=self.stats(100*before.astype(float),draws)
            matrices=[];ece=[];reliability=[]
            for model in range(3):
                matrix=np.zeros((8,8));np.add.at(matrix,(np.broadcast_to(self.y[ids],(repeats,len(ids))).ravel(),predicted[model].ravel()),1/repeats);matrices.append(matrix.tolist())
                bins=np.minimum((confidence[model]*10).astype(int),9);value=0.;points=[]
                for b in range(10):
                    hit=bins==b
                    if hit.any():
                        bin_accuracy=float(accuracy[model][hit].mean());reported=float(confidence[model][hit].mean())
                        value+=hit.mean()*abs(reported-bin_accuracy)
                        points.append(dict(bin=b,count=float(hit.sum()/repeats),accuracy=bin_accuracy,confidence=reported))
                ece.append(float(value));reliability.append(points)
            contrast=None
            if paired:
                pair_draws=self.draws(self.y[ids][::2],seed)
                contrast={key:self.stats(value[:,::2]-value[:,1::2],pair_draws) for key,value in measurements.items()}
            disagreement=float(np.mean([np.mean(predicted[a]!=predicted[b]) for a,b in [(0,1),(0,2),(1,2)]]))
            condition_features=self.features(changed.reshape(-1,28,28,3)).reshape(repeats,len(ids),7)
            variants.append(dict(label=condition['label'],kind=condition.get('kind','none'),level=float(condition.get('level',1)),
                mode=arm,scope=condition.get('scope','whole'),mask_type=condition.get('mask_type','core'),position=condition.get('position','center'),
                pc_offsets=[condition.get('pc1',0),condition.get('pc2',0)],
                probabilities=p.tolist(),reference_probabilities=clean.tolist(),previews=[[self.png(im) for im in row] for row in changed],
                metrics=metrics,ece=ece,reliability=reliability,confusion=matrices,paired_error_minus_control=contrast,
                disagreement=disagreement,doses=doses,features=condition_features.tolist(),
                feature_contrast=self.feature_contrast(condition_features.mean(0),ids,seed) if paired else None))
        used_masks={c.get('mask_type','core') for c in conditions if c.get('scope','whole')!='whole'}
        manual=self.decode_input(request,True) if 'manual' in used_masks else None
        recorded_masks={name:[self.png(np.rint(self.mask(self.images[index],name,manual)*255).astype(np.uint8)) for index in ids] for name in used_masks}
        return dict(schema='bloodmnist-hypothesis-experiment-v1',created_utc=datetime.now(timezone.utc).isoformat(),
            hypothesis=request.get('hypothesis','custom'),requested_design={k:v for k,v in request.items() if k not in ['image','mask','pixels','mask_pixels']},comparison=comparison,seed=seed,repeats=repeats,cohort=cohort,
            matching_model=m if paired else None,paired=paired,unit_count=len(ids)//2 if paired else len(ids),
            ids=ids.tolist(),labels=self.y[ids].tolist(),roles=roles,classes=self.classes,inputs=[self.png(im) for im in inputs],
            variants=variants,input_features=input_features.tolist(),input_feature_contrast=self.feature_contrast(input_features,ids,seed) if paired else None,feature_names=FEATURE_NAMES,model_image_queries=queries,split='test',input_sha256=self.input_sha,
            checkpoint_sha256=self.engine.weight_hashes,models=['resnet18','resnet50','vit'],device=str(self.engine.device),masks=recorded_masks,
            runtime=self.runtime,implementation_sha256=self.implementation_sha,mask_feather_sigma=.7,pc_explained_variance=self.pc_variance,
            uncertainty='500 bootstrap draws; images (pairs kept together), stratified by true class; noise repeats averaged within images. Conditional on selected images/matching; no CI for one independent unit.',
            limits=['Exploratory selected TEST cohort, not independent confirmation or full-population prevalence.',
                    'Original truth labels may cease to apply after edits; ECE is descriptive and unstable for small selected cohorts.',
                    'Automatic/manual ROIs are not expert segmentation; pale cytoplasm/neighbor contamination and unequal dose remain.',
                    'Common resizing changes the clean distribution; ResNet50 always stays native28.',
                    'PCA probes are finite clipped pixel directions with the residual preserved, not global decision boundaries.',
                    'No multiplicity correction; bootstrap intervals exclude training, matching-estimation and mask uncertainty, and can be degenerate with small class strata.'])
