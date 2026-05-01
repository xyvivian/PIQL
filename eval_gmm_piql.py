import os

from dataset_loader.synthetic_data_generator_5000 import DataGenerator
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
import os.path
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import random
from torch import nn

import time
from adbench.myutils import Utils
from data_prior.feature_transform import FeatureTransform
import sklearn.metrics as skm
import re
from tqdm import tqdm

from model_meta_0413 import encoders
from trainer.trainer_performer_0413 import make_model_od




def compute_training_style_ce(logits, labels, num_class=2, ignore_index=-100):
    """Compute CE with the same setup used during training.

    Training uses class weights=[1/num_class, ...], reduction='none', then
    a mean aggregation over positions/batch. Here inference has one eval
    position per sample, so this reduces to a mean over per-sample weighted CE.
    """
    ce_loss_fn = nn.CrossEntropyLoss(
        weight=torch.ones(size=(num_class,)) / num_class,
        reduction='none',
        ignore_index=ignore_index,
    )
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    losses = ce_loss_fn(logits_t, labels_t)
    losses = losses.view(-1, 1)
    return losses.mean(0).mean().item()


def make_model(seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None):
    criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none',
                                    ignore_index=hps['ignore_index'])
    
    model = make_model_od(
        criterion=criterion, 
        encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
                          input_to_internal_encoder_generator = encoders.get_normalized_uniform_seq_encoder(
                              lambda in_dim, out_dim, seq_len, n_tokens: encoders.MLPSeqEncoder(
                                  num_features=in_dim,
                                  emsize=out_dim,
                                  seq_len=seq_len,
                                  n_tokens=n_tokens,
                              )
                          ),
                          internal_encoder_generator = encoders.get_normalized_uniform_encoder(encoders.Linear),
        emsize=emsize, 
        nhead=nhead, 
        nhid=nhid, 
        nlayers=nlayers,
        seq_len=seq_len,
        y_encoder_generator=encoders.Linear,
        extra_prior_kwargs_dict={'num_features': num_features},
        model_para_dict=model_para_dict,
    )
    return model



def run_with_fixed_context(
    model,
    train_x,
    test_x,
    save_path,
    batch_size=5000,
):
    """
    Mini-batched inference with dynamic batch sizing.
    Ensures: train_x + batch_x <= 5000
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    logits = []
    
    d = test_x.shape[2]
    T = test_x.shape[1]
    
    train_size = train_x.shape[0]
    max_batch_allowed = max(1, 5000 - train_size)
    
    model.eval()
    start_time = time.time()
    
    start = 0
    with torch.no_grad():
        while start < T:
            effective_batch_size = min(batch_size, max_batch_allowed, T - start)
            end = start + effective_batch_size
            batch_x = test_x[:, start:end, :].squeeze(0)  # (batch, d)
            batch_x = batch_x.unsqueeze(1)  # (batch, 1, d)
            batch_logits = model(train_x, None, batch_x).squeeze(1)
            logits.append(batch_logits)
            start = end
    
    logits = torch.concatenate(logits, dim=0).cpu().numpy()
    total_time = time.time() - start_time
    
    return logits, total_time







def get_results(model, train_x, test_x, label, save_path, inst_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # train_x: (seq_len-1, 1, d)
    train_y = None
    # test_x: (1[seq-len], num_text_x[batch_size], d[#feat])

    total_time = 0
    with torch.no_grad():
        model.eval()
        # we add our batch dimension, as our transformer always expects that
        logits = []
        for i in tqdm(range(test_x.shape[1])):
            start_time = time.time()
            inst_logits = model(train_x, train_y, test_x[:, i, :].unsqueeze(1))
            # (bs=1, num_test_x=1, num_classes=2)

            duration = time.time() - start_time
            total_time += duration

            inst_logits = inst_logits.squeeze(0)  # squeeze the batch_size=1, get: (num_test_x=1, num_classes=2)
            logits.append(inst_logits)
        logits = torch.concatenate(logits, dim=0).cpu().numpy()
        return logits, total_time


def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies - 1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    FT = FeatureTransform(cfg=cfg)
    train_cfg = cfg.train
    prior_gmm_cfg = cfg.prior.mixture
    test_cfg = cfg.test
    
    seq_len = train_cfg.seq_len
    hyperparameters = train_cfg.hyperparameters
    batch_size = train_cfg.batch_size
    epochs = train_cfg.epochs
    steps_per_epoch = train_cfg.steps_per_epoch
    lr = train_cfg.lr
    device = train_cfg.device
    emsize = train_cfg.emsize
    nhead = train_cfg.nhead
    nhid = train_cfg.nhid
    nlayer = train_cfg.nlayer
    
    num_R = train_cfg.num_R
    reuse_data_every_n = train_cfg.reuse_data_every_n
    gen_one_train_one = train_cfg.gen_one_train_one
    apply_linear_transform = train_cfg.apply_linear_transform
    num_device = train_cfg.num_device
    train_seed = train_cfg.seed
    # prior hyperparameters
    max_feature_dim = prior_gmm_cfg.max_feature_dim
    inflate_full = prior_gmm_cfg.inflate_full
    # test hyperparameters
    feature_truncation = test_cfg.feature_truncation
    seed = test_cfg.seed
    inf_len = test_cfg.inf_len
    preprocess_transform = test_cfg.preprocess_transform
    last_layer_no_R = train_cfg.last_layer_no_R

    print(f'testing with seed={seed}--num_R={num_R}--train_seed={train_seed}')
    
    model_path = '/home/xding2/PIQL/pretrained_ckpts/2layer_piql'
    
    if not os.path.exists(model_path):
        print('file not found:')
        print(model_path)
        raise FileNotFoundError
    # seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None

    trained_model = make_model(seq_len=seq_len, num_features=max_feature_dim, hps=hyperparameters,
                               emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayer,
                               num_class=2, model_para_dict={'num_R': num_R, 'last_layer_no_R':last_layer_no_R})
    trained_model = trained_model.to(device)

    ckpt_files = [f for f in os.listdir(model_path) if f.endswith('.ckpt')]
    ckpt_infos = []
    epoch_pat = re.compile(r"epoch=(\d+)")
    for filename in ckpt_files:
        m = epoch_pat.search(filename)
        if m is None:
            print(f'Skipping checkpoint without epoch=XX pattern: {filename}')
            continue
        ckpt = int(m.group(1))
        ckpt_infos.append((ckpt, filename))

    ckpt_infos.sort(key=lambda x: x[0])

    for ckpt, filename in ckpt_infos:
        ckpt_path = os.path.join(model_path, filename)
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Extract the state_dict
        state_dict = checkpoint['state_dict']

        # Remove the 'model.' prefix from the keys (if any) saved by pytorch_lightning
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = state_dict[key]
        trained_model.load_state_dict(new_state_dict)

        print(f'loading from {ckpt_path}')

        print(
            f"Using a Transformer with {sum(p.numel() for p in trained_model.parameters()) / 1000 / 1000:.{2}f} M parameters")
        setting = 'semi' #'semi'
        utils = Utils()  # utils function
        utils.set_seed(seed)
        datagenerator = DataGenerator(seed=seed)  # data generator
        
        utils = Utils()  # utils function
        utils.set_seed(seed)
        # Get the datasets from synthetic data
        for dataset_list in [datagenerator.dataset_list_gmm]:
            data_name = dataset_list[0].split('_')[0]
            for dataset in dataset_list:
                '''
                la: ratio of labeled anomalies, from 0.0 to 1.0
                realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
                noise_type: inject data noises for testing prior robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
                '''
                print(dataset)

                identifier = f'train_seed={train_seed}.transform={preprocess_transform}.feat_truncation={feature_truncation}'
                if test_cfg.extra_suffix != '':
                    identifier = identifier + f'.{test_cfg.extra_suffix}'

                model_name = f'piql_{ckpt}'  # 'fomo'  #'fomoorigin2'  #'gmm'  #'fomoorigin2' #'fomo'
                save_path = f'results/gaussian_5000/2layer_piql/{model_name}/{dataset}'
                print(save_path)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                datagenerator.dataset = dataset  # specify the dataset name
                data = datagenerator.generator(la=0, max_size=50000, scale=False)  # maximum of 50,000 data points are available

                def add_feature_transform(x, eval_position):
                    if preprocess_transform is None:
                        feature_dim = x.shape[-1]
                        if feature_dim > max_feature_dim:
                            if feature_truncation == 'projection':
                                x = FT.feature_sparse_projection(x=x, num_feature=feature_dim)
                            else:
                                x = FT.feature_subsampling(x=x, num_feature=feature_dim)
                        if feature_dim < max_feature_dim:
                            x = FT.feature_padding(x=x, num_feature=feature_dim)
                    else:
                        x = FT.pfn_inference_transform(eval_xs=x, preprocess_transform=preprocess_transform,
                                                    eval_position=eval_position,
                                                    normalize_with_test=True, rescale_with_sqrt=False)
                    return x



                def make_train_test(data):
                    x_train = data['X_train']
                    y_train = data['y_train']
                    x_test = data['X_test']
                    y_test = data['y_test']
                    mean_train = data['mean_train']
                    variance_train = data['variance_train']
                    mean_test = data['mean_test']
                    variance_test = data['variance_test']
                    global_mean = data['global_mean']
                    global_variance = data['global_variance']
                    global_anomaly_mean = data['global_anomaly_mean']
                    global_anomaly_variance = data['global_anomaly_variance']

                    if x_train.shape[0] <= inf_len - 1:
                        train_x = x_train
                    else:
                        train_sub_indices = np.random.choice(x_train.shape[0], inf_len - 1, replace=False)
                        train_x = x_train[train_sub_indices]

                    train_and_test = add_feature_transform(x=np.concatenate([train_x, x_test], axis=0),
                                                        eval_position=len(train_x))

                    train_x, x_test = train_and_test[:len(train_x), :], train_and_test[len(train_x):, :]

                    test_in = x_test[y_test == 0]  # #inst, d
                    test_out = x_test[y_test == 1]  # #inst, d

                    train_x = torch.from_numpy(train_x).to(device).unsqueeze(1).float()
                    test_in = torch.from_numpy(test_in).to(device).unsqueeze(0).float()
                    test_out = torch.from_numpy(test_out).to(device).unsqueeze(0).float()

                    print('train_x shape:', train_x.shape)
                    print('test_in shape', test_in.shape)
                    print('test_out  shape', test_out.shape)
                    return train_x, test_in, test_out

                train_x, test_in, test_out = make_train_test(data)
                logits_in, time_in = run_with_fixed_context(model=trained_model, train_x=train_x, test_x=torch.cat([test_in,test_out],dim=1),
                                                        save_path=save_path + '/seed{}'.format(seed),batch_size=5000)
                logits_in = logits_in[0:test_in.shape[1] + test_out.shape[1],:]
                labels = np.array([0] * test_in.shape[1] + [1] * test_out.shape[1])
                exp_logits = np.exp(logits_in)
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                # compute CE in a way comparable to training
                ce_loss = compute_training_style_ce(
                    logits=logits_in,
                    labels=labels,
                    num_class=2,
                    ignore_index=hyperparameters['ignore_index'],
                )
                print(f"train_comparable_ce_loss: {ce_loss:.6f}")

                score = probabilities[:, 1]
                indices = np.arange(len(labels))
                p = low_density_anomalies(-score, len(indices[labels == 1]))
                f1_score = skm.f1_score(labels, p)
                inds = np.where(np.isnan(score))
                score[inds] = 0
                result = utils.metric(y_true=labels, y_score=score)

                result = {seed: {'aucroc': result['aucroc'], 'aucpr': result['aucpr'], 'f1': f1_score,
                                 'ce_loss': ce_loss,
                                'time-per-inst': time_in / len(labels),
                                'context-len': train_x.shape[0]}}

                df = pd.DataFrame(result)
                df = df.transpose().reset_index()
                df.columns = ['seed', 'aucroc', 'aucpr', 'f1', 'ce_loss', 'time-per-inst', 'context-len']
                df.to_csv(save_path + '/result.csv', mode='a', index=False)

def run_with_overrides(base_overrides, overrides):
    GlobalHydra.instance().clear()
    initialize(config_path="configs")
    cfg = compose(config_name="config", overrides=base_overrides + overrides)
    main(cfg)

if __name__ == "__main__":
    override_configs = [[f"test.seed={0}"]] 
    import sys
    base_overrides = sys.argv[1:]

    for overrides in override_configs:
        run_with_overrides(base_overrides, overrides)



#CUDA_VISIBLE_DEVICES=0 python eval_gmm_piql.py train.apply_linear_transform=False train.gen_one_train_one=True prior.mixture.max_feature_dim=100 prior.mixture.max_model_dim=100 train.seed=0 train.num_R=500 test.seed=0 test.preprocess_transform=null


#larger model
#CUDA_VISIBLE_DEVICES=0 python eval_gmm_piql.py train.apply_linear_transform=False train.gen_one_train_one=True prior.mixture.max_feature_dim=240 prior.mixture.max_model_dim=240 train.seed=0 train.num_R=500 test.seed=0 test.preprocess_transform=null