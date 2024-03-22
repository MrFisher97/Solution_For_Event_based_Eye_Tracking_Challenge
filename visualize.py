import h5py
import numpy as np
import os,json
import os.path as osp
import tonic.transforms as transforms

import dataset.ThreeET_plus as Dataset
import dataset.custom_transforms as T
from torch.utils.data import DataLoader
import torch
import csv, cv2, tqdm
from imageio import mimsave
from utils.metrics import p_acc, px_euclidean_dist
import importlib
import yaml
    
def main(args):
    with open('configs/' + args.config_file) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    args = argparse.Namespace(**config)

    # test data loader always cuts the event stream with the labeling frequency
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    label_transform = transforms.Compose([
        T.ScaleLabel(factor),
        T.TemporalSubsample(temp_subsample_factor),
        T.NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    split = 'val'
    if split == 'val':
        lenth = args.val_length
        stride = args.val_stride
    elif split == 'test':
        lenth = args.test_length
        stride = args.test_stride

    slicing_time_window = lenth*int(10000/temp_subsample_factor) #microseconds
    stride_time = int(10000/temp_subsample_factor*stride) #microseconds
    data_orig = Dataset.MyDataset(save_to=args.data_dir, 
                                  split=split, 
                                  transform=transforms.Downsample(spatial_factor=factor),
                                  target_transform=label_transform)
    
    slicer = T.SliceByTimeEventsTargets(slicing_time_window, 
                                    overlap=slicing_time_window-stride_time, 
                                    seq_length=lenth, 
                                    seq_stride=stride, 
                                    include_incomplete=False)
    post_slicer_transform = transforms.Compose([
        T.SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        T.EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2),
                                n_time_bins=args.n_time_bins, 
                                per_channel_normalize=args.voxel_grid_ch_normaization,
                                map_type='binary')
    ])
    dataset = Dataset.MySlicedDataset(data_orig, slicer, transform=post_slicer_transform, metadata_path=None)

    args.batch_size = 1 
    # otherwise the collate function will through an error. 
    # This is only used in combination of include_incomplete=True during testing
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=2)

    # Define your model, optimizer, and criterion
    model = importlib.import_module(f"model.{args.model}").Model(args).to(args.device)
    # load weights from a checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        raise ValueError("Please provide a checkpoint file.")
    
    output_path = f"{osp.dirname(args.checkpoint)}/{osp.basename(args.checkpoint).split('.')[0].split('_')[2]}/{split}"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(osp.join(output_path, 'gif'), exist_ok=True)
    
    # evaluate on the validation set and save the predictions into a csv file.
    with open(osp.join(output_path, f"predict.csv"), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        # add column names 'row_id', 'x', 'y'
        csv_writer.writerow(['row_id', 'x', 'y'])
        row_id = 0
        p10 = []
        p_error = []                
        tmps = []
        cur_index = None
        for batch_idx, (data, target, data_name, _) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            if cur_index == None:
                cur_index = data_name[0]
                cur_p10, cur_err, cur_size, cur_len = [], [], [], []
                occ_time = torch.zeros([10000, 2], device=args.device)
                output_list = torch.zeros([10000, 2], device=args.device)
                cur_loc = 0
            elif (cur_index != data_name[0]) or (batch_idx == (len(dataset) - 1)):
                mimsave(osp.join(output_path, 'gif', f"{cur_index}_{sum(cur_p10)/sum(cur_size):.3f}_{sum(cur_err)/sum(cur_len):.2}.gif"), 
                        tmps, 
                        fps=10)
                tmps = []
                cur_p10, cur_err, cur_size, cur_len = [], [], [], []
                cur_index = data_name[0]
                occ_time = torch.zeros([10000, 2], device=args.device)
                output_list = torch.zeros([10000, 2], device=args.device)
                cur_loc = 0
            
            data = data.to(args.device)
            output = model(data)

            output, target = output[0], target[0]
            output = output.clip(0, 0.999)

            occ_time[cur_loc:cur_loc+output.size(0)] += 1
            output_list[cur_loc:cur_loc+output.size(0)] += output
            
            output = output_list[cur_loc:cur_loc+args.val_stride]
            output /= occ_time[cur_loc:cur_loc+args.val_stride]
            target = target[:args.val_stride]
            cur_loc += args.val_stride

            # calculate the metric
            p_corr, batch_size = p_acc(target[:,:2], 
                                       output.to(target), 
                                       width_scale=640*factor, 
                                       height_scale=480*factor, 
                                       pixel_tolerances=[10])
            
            p_error_total, bs_times_seqlen = px_euclidean_dist(target, 
                                                               output.to(target),
                                                               width_scale=640*factor, 
                                                               height_scale=480*factor)
            
            p10.append(p_corr['p10'].item()/batch_size)
            p_error.append(p_error_total.item()/bs_times_seqlen)

            # Important! 
            # cast the output back to the downsampled sensor space (80x60)
            output = output * torch.tensor((640*factor, 480*factor)).to(args.device)

            for frame_id in range(target.shape[0]):
                img = data[0, frame_id].abs().cpu().numpy()
                img = img.transpose(1,2,0)
                img = np.concatenate([img, torch.zeros(img.shape[0], img.shape[1], 1)], axis=2)
                img = 1 - img
                img *= 255.
                
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
                
                x_gt = round(target[frame_id, 0].item()*args.sensor_width)
                y_gt = round(target[frame_id, 1].item()*args.sensor_height)
                x_pred = round(output[frame_id, 0].item()/factor)
                y_pred = round(output[frame_id, 1].item()/factor)

                cv2.circle(img, (x_gt, y_gt), 5, (255, 0, 0), -1)
                cv2.circle(img, (x_gt, y_gt), int(10/factor), (255, 0, 0), 1)
                cv2.circle(img, (x_pred, y_pred), 5, (0, 0, 0), -1)
                tmps.append(img.astype(np.uint8))
                row_to_write = output[frame_id].tolist()
                # prepend the row_id
                row_to_write.insert(0, row_id)
                csv_writer.writerow(row_to_write)
                row_id += 1

            cur_p10.append(p_corr['p10'].item())
            cur_err.append(p_error_total.item())
            cur_size.append(batch_size)
            cur_len.append(bs_times_seqlen)
        print(f"Model [{args.model}] avg p10: {np.array(p10).mean():.3f}, p_error: {np.array(p_error).mean():.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", 
                        default="task.yaml", 
                        help="path to YAML configuration file")
    
    parser.add_argument("--checkpoint", 
                        default="output/CNN_GRU_base/240316014519/model_best_ep679_val_loss_0.0391.pth")

    args = parser.parse_args()
    main(args)