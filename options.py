gpu = "0"
random_seed = 0
data_type = "coords"
video_path = "lip_images/"
train_list = f"data/{data_type}_train.txt"
val_list = f"data/{data_type}_val.txt"
anno_path = "GRID_alignments"
coords_path = "lip_coordinates"
vid_padding = 75
txt_padding = 200
batch_size = 40
base_lr = 2e-5
num_workers = 16
max_epoch = 350
display = 50
test_step = 1000
model_prefix = "ResNet_Coord_BiLSTM_NoTransformer"
save_prefix = f"weights/ResNet+Coord+BiGRU+transformer_new_{model_prefix}_{data_type}"
is_optimize = True
pin_memory = True

#weights = "weights/not_pretrained_5_LipCoordNet_coords_at_190+70+10+90+epoch_110_loss_0.2825022041797638_wer_0.34291666666666665_cer_0.10250950239049371.pt"
#weights = "weights/LipCoordNet_with_transformer_coords_at_470+new_10_epoch_499_loss_1.7897634506225586_wer_1.0000833333333332_cer_0.6124907981586577.pt"

#marco updated this
# weights = "weights/LipCoordNet_with_transformer_coords_at_470+new_10_epoch_410_loss_0.03242040053009987_wer_0.03658333333333333_cer_0.013786586554791203.pt""
#weights = "weights/LipNet2_coords_at_470+new_10_epoch_149_loss_2.269651412963867_wer_0.9939166666666668_cer_0.7859287154366854.pt"
