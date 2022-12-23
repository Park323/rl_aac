import torch
import torch.nn.functional as F
from .eval_metrics import evaluate_metrics_from_lists, combine_single_and_per_file_metrics, write_json
from .cider.cider import Cider

def CIDEr(outputs, tokenizer):
    gt_captions = {}
    pred_captions = {}
    for i_ex in range(outputs['predictions'].shape[0]):
        # gt_ = tokenizer.batch_decode(outputs['label_ids'][i_ex,:])
        gt_ = outputs['label_ids'][i_ex,:]
        gt = [decode_to_string(gt__) for gt__ in gt_]
        # gt = [refine_decoded_string(gt__) for gt__ in gt_]
        # pred_ = tokenizer.decode(outputs['predictions'][i_ex,:])
        pred_ = outputs['predictions'][i_ex,:]
        # pred = [refine_decoded_string(pred_)]
        pred = [decode_to_string(pred_)]
        gt_captions[f'{i_ex}'] = gt
        pred_captions[f'{i_ex}'] = pred

    scorer = Cider()
    _, scores = scorer.compute_score(gt_captions, pred_captions)
    return scores


def decode_to_string(ids):
    return " ".join(map(str, ids.cpu().tolist()))

def refine_decoded_string(string):
    return string.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', '')

def aac_metrics(outputs, tokenizer):
    all_gt_captions = []
    all_pred_captions = []
    
    gt_captions = []
    pred_captions = []
    for i_ex in range(outputs['predictions'].shape[0]):
        gt_ = tokenizer.decode(outputs['label_ids'][i_ex,:])
        gt_captions.append(gt_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))
        pred_ = tokenizer.decode(outputs['predictions'][i_ex,:])
        pred_captions.append(pred_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))
        
        # Group for COCO metrics
        if i_ex == len(outputs['filenames'])-1 or outputs['filenames'][i_ex+1] != outputs['filenames'][i_ex]: # Last example for current audio
            assert(all(x == pred_captions[0] for x in pred_captions))
            all_gt_captions.append(gt_captions)
            all_pred_captions.append(pred_captions[0])
            
            # print('----------')
            # print('Pred: '+pred_captions[0])
            # print('GTs:')
            # for i_gt in range(len(gt_captions)):
            #    print('      '+gt_captions[i_gt])
            
            gt_captions = []
            pred_captions = []
    
    metrics, per_file_metrics = evaluate_metrics_from_lists(all_pred_captions, all_gt_captions)
    
    file_names = ['{:05d}'.format(i_file) for i_file in range(len(all_gt_captions))]
    
    total_metrics = combine_single_and_per_file_metrics(
        metrics, per_file_metrics, file_names
    )
    
    return {
        key.lower(): value for key, value in total_metrics.items()
    }, all_gt_captions, all_pred_captions


def cosine_similarity(output, target):
    return F.cosine_similarity(output, target).mean()


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
