import yaml

if __name__ == '__main__':

    entries = [
        {
            'team': 'jiaxingdns',
            'model': 'spikingdns',
            'date': '2023-08-18',
            'SI-SNR': 14.11,
            'SI-SNRi_data': 14.11 - 7.62,
            'SI-SNRi_enc+dec': 14.11 - 7.62,
            'MOS_ovrl': 2.77,
            'MOS_sig': 3.16,
            'MOS_bak': 3.65,
            'latency_enc+dec_ms': 0.008,
            'latency_total_ms': 8.009,
            'power_proxy_Ops/s': None,
            'PDP_proxy_Ops': None,
            'params': 793 * 10**3,
            'size_kilobytes': None,
            'model_path': 'checkpoints/Traineds4_snn_plif_final/ckpt_epoch49.pt',
        },
        ]
      
    with open('./metricsboard_track_1_validation.yml', 'w') as outfile:
        yaml.dump(entries, outfile)