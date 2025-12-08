This folder includes dataset for training and testing.

## Data Format:

"channel": wide beam received signal      format: (2, 4096, 10, 16) which means (Real/Imaginary part, Data number,Time slot, Wide beam index)

"max_id_sery_no_noise_m": optimal narrow beam index   format: (4096, 10) which means (Data number, Time slot)

"rsrp_sery_no_noise_m": narrow beam received power 	format:(4096, 10, 64) which means (Data number,Time slot, Narrow beam index)
