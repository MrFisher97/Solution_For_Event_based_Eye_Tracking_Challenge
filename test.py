"""
Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import argparse, os, csv,tqdm
import torch
from torch.utils.data import DataLoader

import importlib
import dataset.ThreeET_plus as Dataset
import dataset.custom_transforms as T
import tonic.transforms as transforms
import importlib

import yaml

def test(args, model):
    # test data loader always cuts the event stream with the labeling frequency
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    label_transform = transforms.Compose([
        T.ScaleLabel(factor),
        T.TemporalSubsample(temp_subsample_factor),
        T.NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    test_data_orig = Dataset.MyDataset(save_to=args.data_dir, split='test', \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    slicing_time_window = args.test_length*int(10000/temp_subsample_factor) #microseconds
    stride_time = int(10000/temp_subsample_factor*args.test_stride) #microseconds

    test_slicer = T.SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-stride_time, \
                    seq_length=args.test_length, seq_stride=args.test_stride, include_incomplete=True)

    post_slicer_transform = transforms.Compose([
        T.SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        T.EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization,
                                map_type='binary'),
    ])

    test_data = Dataset.MySlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform)

    args.batch_size = 1 
    # otherwise the collate function will through an error. 
    # This is only used in combination of include_incomplete=True during testing
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, \
                            num_workers=2)
    
    model.eval()
    # evaluate on the validation set and save the predictions into a csv file.
    with open(os.path.join(args.log_dir, 'sample_submission.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        # add column names 'row_id', 'x', 'y'
        csv_writer.writerow(['row_id', 'x', 'y'])
        row_id = 0
        cur_index = None
        for batch_idx, (data, target_placeholder, data_name, last_inp) in tqdm.tqdm(enumerate(test_loader)):
            if (cur_index == None) or (cur_index != data_name[0]) or (batch_idx == (len(test_data) - 1)):
                cur_index = data_name[0]
                occ_time = torch.zeros([10000, 2], device=args.device)
                output_list = torch.zeros([10000, 2], device=args.device)
                cur_loc = 0
            data = data.to(args.device)
            output = model(data)

            output, target_placeholder = output[0], target_placeholder[0]
            output = output.clip(0, 0.999)
            
            occ_time[cur_loc:cur_loc+output.size(0)] += 1
            output_list[cur_loc:cur_loc+output.size(0)] += output

            if last_inp:
                end_len = output.size(0)
                model.hidden = None
            else:
                end_len = args.test_stride
            output = output_list[cur_loc:cur_loc+end_len]
            output /= occ_time[cur_loc:cur_loc+end_len]
            target_placeholder = target_placeholder[:end_len]
            cur_loc += end_len
            # Important! 
            # cast the output back to the downsampled sensor space (80x60)
            output = output * torch.tensor((640*factor, 480*factor)).to(args.device)

            for frame_id in range(target_placeholder.shape[0]):
                row_to_write = output[frame_id].tolist()
                # prepend the row_id
                row_to_write.insert(0, row_id)
                csv_writer.writerow(row_to_write)
                row_id += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # a config file 
    parser.add_argument("--config_file", 
                        default="task_local.yaml", 
                        help="path to YAML configuration file")
    # load weights from a checkpoint
    parser.add_argument("--checkpoint",  
                        default="output/CNN_GRU/240316014519/model_best_ep679_val_loss_0.0391.pth")
    parser.add_argument("--log_dir", type=str, default='./')

    args = parser.parse_args()

    # Load hyperparameters from YAML configuration file
    with open('configs/' + args.config_file) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)['hyperparameters']
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    args = argparse.Namespace(**config)

    # Define your model, optimizer, and criterion
    model = importlib.import_module(f"model.{args.model}").Model(args).to(args.device)
    # load weights from a checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        raise ValueError("Please provide a checkpoint file.")

    test(args, model)
