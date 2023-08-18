import yaml

if __name__ == '__main__':

    entries = [
        {
            'team': 'jiaxingdns',
            'model': 'spikingdns',
            'date': '2023-02-20',
            'SI-SNR': 12.50,
            'SI-SNRi_data': 12.50 - 7.62,
            'SI-SNRi_enc+dec': 12.50 - 7.62,
            'MOS_ovrl': 2.71,
            'MOS_sig': 3.21,
            'MOS_bak': 3.46,
            'latency_enc+dec_ms': 0.036,
            'latency_total_ms': 8.036,
            'power_proxy_Ops/s': 11.59 * 10**6,
            'PDP_proxy_Ops': 0.09 * 10**6,
            'params': 793 * 10**3,
            'size_kilobytes': 465,
            'model_path': 'checkpoints/Traineds4_snn_plif_final/ckpt_epoch49.pt',
        },
        ]
      
    with open('./metricsboard_track_1_validation.yml', 'w') as outfile:
        yaml.dump(entries, outfile)