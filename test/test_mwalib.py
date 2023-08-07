from pymwalib.metafits_context import MetafitsContext
from pymwalib.voltage_context import VoltageContext
import glob

metafits = "/datax2/users/dancpr/2023.jun-tian-bifrost-mwa/1369756816.metafits"
p = sorted(glob.glob('/datax2/users/dancpr/2023.jun-tian-bifrost-mwa/1369756816_*.sub'))
coarse_chan = 0

vcs = VoltageContext(metafits, p)
meta  = vcs.metafits_context
coarse_channel_idx = coarse_chan

obs_len = vcs.common_duration_ms / 1e3
N_frame    = int(obs_len)  # 1 second frames

N_coarse_chan     = 1 #self.vcs.num_coarse_chans     # Only load one coarse channel at a time
N_station         = vcs.metafits_context.num_ants
N_pol             = vcs.metafits_context.num_ant_pols
N_samp            = vcs.num_samples_per_voltage_block
N_block           = vcs.num_voltage_blocks_per_second
N_cplx            = 2

d_shape = (N_block, N_coarse_chan, N_station, N_pol, N_samp, 2)
t0_gps = int(vcs.timesteps[0].gps_time_ms / 1e3)

frame_count = 0

for frame_count in range(N_frame):
    d = vcs.read_second(t0_gps + frame_count, 1, coarse_channel_idx).reshape(d_shape)
    print(frame_count)