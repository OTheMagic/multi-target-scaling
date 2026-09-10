"""Original fresh-data DGP and training-only fitted model, copied without changes."""
import numpy as np
def generate(seed, family, n, d=6, train=800, test=1200, base_alpha=.1):
    rng = np.random.default_rng(seed)
    beta = np.sin(np.arange(3*d).reshape(3,d)+1)/2
    scales = np.geomspace(.6, 2.4, d)
    arrays = {}
    for split, count in [('train',train), ('cal',n), ('test',test)]:
        x = rng.normal(size=(count,3))
        group = (x[:,0] > 0).astype(int)
        if family in ('gaussian','correlated'):
            noise = rng.normal(size=(count,d))
            if family == 'correlated':
                noise = np.sqrt(.2)*noise + np.sqrt(.8)*rng.normal(size=(count,1))
        elif family == 'laplace':
            noise = rng.laplace(size=(count,d))/np.sqrt(2)
        elif family == 'skewed':
            noise = (np.exp(.8*rng.normal(size=(count,d))) - np.exp(.8**2/2))
            noise /= np.sqrt((np.exp(.8**2)-1)*np.exp(.8**2))
        else:
            raise ValueError(family)
        y = x@beta + (1+.8*group[:,None])*scales*noise
        arrays['X_'+split], arrays['Y_'+split], arrays['group_'+split] = x,y,group
    xt = np.column_stack([np.ones(train), arrays['X_train']])
    fitted = np.linalg.lstsq(xt, arrays['Y_train'], rcond=None)[0]
    train_error = arrays['Y_train']-xt@fitted
    quantiles = np.array([np.quantile(train_error[arrays['group_train']==g],
                                     [base_alpha/2, 1-base_alpha/2], axis=0)
                          for g in [0,1]])
    arrays.update(model_coef=fitted, fitted_error_quantiles=quantiles,
                  beta=beta, scales=scales)
    for split in ['cal','test']:
        pred = np.column_stack([np.ones(len(arrays['X_'+split])),arrays['X_'+split]])@fitted
        group = arrays['group_'+split]
        lo,hi = pred+quantiles[group,0],pred+quantiles[group,1]
        raw = np.maximum(lo-arrays['Y_'+split],arrays['Y_'+split]-hi)
        arrays.update({f'pred_{split}':pred, f'lo_{split}':lo, f'hi_{split}':hi,
                       f'abs_{split}':abs(arrays['Y_'+split]-pred),
                       f'raw_{split}':raw, f'width_{split}':hi-lo})
    return arrays


