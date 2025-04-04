import os,sys
import time
import torch
import argparse

from model import SASRec
from utils import *
import geometry
from torch.optim.lr_scheduler import CosineAnnealingLR
import geoopt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.dirname( __file__ )))


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)





def SASRec_exp(dataset = 'ml-1m', pos_lambda_man_reg = 0,neg_lambda_man_reg = 0, geoopt_emb = False, num_epochs = 2000, maxlen = 200, hidden_units = 50):
    args = parser.parse_args(['--dataset', dataset, '--train_dir','time_{}_lambda_pos_{}_lambda_neg_{}_geoopt_{}'.format(int(time.time()),pos_lambda_man_reg,neg_lambda_man_reg,geoopt_emb),'--num_epochs', str(num_epochs)
                              ,'--maxlen',str(maxlen),'--hidden_units', str(hidden_units)])

    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()


    u2i_index, i2u_index = build_index(args.dataset)
    
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args, geoopt_emb=geoopt_emb).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    #model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    if geoopt_emb:
       adam_optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
    #scheduler = CosineAnnealingLR(adam_optimizer, T_max=args.num_epochs)


    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()

    reg_loss = torch.tensor(0)
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        loss_list = []
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            if pos_lambda_man_reg > 0:
                pos_eseq = model.item_emb(torch.tensor(pos).to('cuda'))
                pos_affinity = geometry.pairseq_dist_affinity(pos_eseq,geometry.hyperbolic_dist)

                pos_logits_dist = torch.cdist(pos_logits,pos_logits)**2

                pos_lap = pos_affinity*pos_logits_dist

                pos_man_reg = pos_lap.sum()
            else:
                pos_man_reg=torch.tensor(0)

            if neg_lambda_man_reg > 0:

                neg_eseq = model.item_emb(torch.tensor(neg).to('cuda'))

                neg_affinity = geometry.pairseq_dist_affinity(neg_eseq,geometry.hyperbolic_dist)

                neg_logits_dist = torch.cdist(neg_logits,neg_logits)**2

                neg_lap = neg_affinity*neg_logits_dist

                neg_man_reg = neg_lap.sum()
            else:
                neg_man_reg=torch.tensor(0)


            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): bce_loss += args.l2_emb * torch.norm(param)


            # # pos_emb has shape [batch_size, max_item, emb_size]
            # pos_tensor = torch.tensor(pos, device=args.device)  # Shape: [batch_size, max_item]

            # # Mask out padding values (if 0 is padding)
            # valid_mask = pos_tensor > 0  # Shape: [batch_size, max_item] (True for valid items)

            # # Get embeddings while preserving batch structure
            # pos_emb = model.item_emb(pos_tensor)  # Shape: [batch_size, max_item, emb_size]

            # # Compute the manifold regularization loss per sequence
            # batch_reg_loss = []
            # for i in range(pos_emb.shape[0]):  # Iterate over batch dimension
            #     valid_items = pos_emb[i][valid_mask[i]]  # Extract only valid item embeddings
            #     if valid_items.shape[0] > 1:  # Ensure at least 2 items for distance calculation
            #         reg_loss = geometry.manifold_regularization_embeddings(valid_items, metric='hyperbolic',sigma=1, lambda_reg=0.01)
            #         batch_reg_loss.append(reg_loss)

            # reg_loss = torch.stack(batch_reg_loss).mean()

            # loss = bce_loss + reg_loss

            loss = bce_loss

            if pos_lambda_man_reg > 0:
                loss += pos_lambda_man_reg*pos_man_reg
            if neg_lambda_man_reg > 0:
                loss += neg_lambda_man_reg*neg_man_reg

            loss.backward()
            adam_optimizer.step()
            if geoopt_emb:
                with torch.no_grad():
                    model.item_emb.embeddings.data = model.item_emb.manifold.projx(model.item_emb.embeddings.data)
            #print("loss in epoch {}  iteration {}: {:.4f} graph_loss_pos {:.4f} graph_loss_neg {:.4f}".format(epoch, step, loss.item(), pos_lambda_man_reg*pos_man_reg.item(),neg_lambda_man_reg*neg_man_reg.item())) # expected 0.4~0.6 after init few epochs
            loss_list.append(loss.item())
            #print("loss in epoch {}  iteration {}: {:.4f} graph_loss {:.4f}".format(epoch, step, bce_loss.item(), reg_loss.item())) # expected 0.4~0.6 after init few epochs
        #scheduler.step()
        print("mean loss in epoch {}: {:.4f} graph_loss_pos {:.4f} graph_loss_neg {:.4f}".format(epoch, torch.mean(torch.tensor(loss_list)).item(), pos_lambda_man_reg*pos_man_reg.item(),neg_lambda_man_reg*neg_man_reg.item())) # expected 0.4~0.6 after init few epochs

        if (epoch % 20 == 0) or (epoch == args.num_epochs):
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            valid_string = 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'% (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            print(valid_string)

            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()


if __name__ == '__main__':

    # for i in range(10):
    #     SASRec_exp(dataset = 'ml-1m', pos_lambda_man_reg = 0,neg_lambda_man_reg = 0, geoopt_emb = False)
    #     SASRec_exp(dataset = 'ml-1m', pos_lambda_man_reg = 0.1,neg_lambda_man_reg = 0.5, geoopt_emb = False)
    #     SASRec_exp(dataset = 'ml-1m', pos_lambda_man_reg = 0,neg_lambda_man_reg = 0, geoopt_emb = True)
    for i in range(10):
        SASRec_exp(dataset = 'Video', pos_lambda_man_reg = 0,neg_lambda_man_reg = 0, geoopt_emb = False, num_epochs = 20, maxlen=10,hidden_units=5)
        SASRec_exp(dataset = 'Video', pos_lambda_man_reg = 0.01,neg_lambda_man_reg = 0.05, geoopt_emb = False, num_epochs=20, maxlen=10,hidden_units=5)
        SASRec_exp(dataset = 'Video', pos_lambda_man_reg = 0,neg_lambda_man_reg = 0, geoopt_emb = True,num_epochs=20, maxlen=10,hidden_units=5)



    print("Done")
